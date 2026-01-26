# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, compute_teacher=False):
        # Determine which data to use based on compute_teacher flag
        if compute_teacher:
            response_length = micro_batch["teacher_response"].size(-1)
        else:
            response_length = micro_batch["responses"].size(-1)
        
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if compute_teacher:
                input_ids = micro_batch["teacher_input_ids"]
                batch, seqlen = input_ids.shape
                attention_mask = micro_batch["teacher_attention_mask"]
                position_ids = micro_batch["teacher_position_ids"]
            else:
                input_ids = micro_batch["input_ids"]
                batch, seqlen = input_ids.shape
                attention_mask = micro_batch["attention_mask"]
                position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    # For AutoModelForTokenClassification with num_labels=1
                    # output.logits shape: (1, total_nnz, 1) or (1, total_nnz)
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz) or (total_nnz, 1)
                    if values_rmpad.dim() == 2:
                        values_rmpad = values_rmpad.squeeze(-1)  # (total_nnz)
                    values_rmpad = values_rmpad.unsqueeze(-1)  # (total_nnz, 1) for pad_input

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                # For sequence-level reward model: extract current token values
                values = values[:, -response_length:]
                
                # Apply last token mask for sequence-level scoring
                response_mask = attention_mask[:, -response_length:]
                response_lengths = response_mask.sum(dim=1).long()
                last_token_indices = response_lengths - 1
                last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
                last_token_mask[batch_indices, last_token_indices] = True
                values = values * last_token_mask.type_as(values)
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    # For AutoModelForTokenClassification with num_labels=1
                    # output.logits shape: (batch, seq_len, 1) or (batch, seq_len)
                    values = output.logits
                # Squeeze the last dimension if num_labels=1
                values = values[:, -response_length:].squeeze(-1)  # (batch, response_length)
                
                # Apply last token mask for sequence-level scoring
                response_mask = attention_mask[:, -response_length:]
                response_lengths = response_mask.sum(dim=1).long()
                last_token_indices = response_lengths - 1
                last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
                last_token_mask[batch_indices, last_token_indices] = True
                values = values * last_token_mask.type_as(values)
            return values

    def _forward_batch_teacher_forcing_grpo(self, batch, teacher_repeat):
        """
        Teacher forcing for GRPO: assign incremental values to teacher responses in the same group.
        
        Args:
            batch: Batch containing teacher data
            teacher_repeat: Number of teacher responses per prompt
        
        Returns:
            values: Tensor with teacher forcing values
        """
        response_length = batch["teacher_response"].size(-1)
        input_ids = batch["teacher_input_ids"]
        bsz, seqlen = input_ids.shape
        attention_mask = batch["teacher_attention_mask"]
        
        values = torch.zeros((bsz, response_length), device=input_ids.device)
        response_mask = attention_mask[:, -response_length:]
        response_lengths = response_mask.sum(dim=1).long()
        last_token_indices = response_lengths - 1
        
        # Assign incremental values for teacher responses in the same group
        for i in range(0, bsz, teacher_repeat):
            for j in range(teacher_repeat):
                values[i + j, last_token_indices[i + j]] = float(j)
        
        return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        # Check if computing teacher values
        compute_teacher = data.meta_info.get("compute_teacher", False)
        
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        
        # Select keys based on compute_teacher flag
        if compute_teacher:
            select_keys = ["teacher_response", "teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"]
        else:
            select_keys = (
                ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
                if "response_mask" in data.batch
                else ["responses", "input_ids", "attention_mask", "position_ids"]
            )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        
        # Teacher forcing for GRPO
        if compute_teacher and "teacher_repeat" in data.meta_info:
            teacher_repeat = data.meta_info["teacher_repeat"]
            batch = data.select(batch_keys=select_keys).batch
            return self._forward_batch_teacher_forcing_grpo(batch, teacher_repeat=teacher_repeat)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        values_lst = []
        for i, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                values = self._forward_micro_batch(model_inputs, compute_teacher=compute_teacher)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        # Apply response mask (already applied in _forward_micro_batch for GAD mode)
        if not compute_teacher and "response_mask" in data.batch:
            response_mask = data.batch["response_mask"]
            response_mask = response_mask.to(values.device)
            values = values * response_mask  # Only action tokens have values
        elif compute_teacher:
            # For teacher values, apply teacher response mask
            responses = data.batch.get("teacher_response")
            if responses is not None:
                attention_mask = data.batch.get("teacher_attention_mask")
                if attention_mask is not None:
                    response_length = responses.size(1)
                    response_mask = attention_mask[:, -response_length:]
                    values = values * response_mask
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}
        
        # Check if using GAD discriminator training
        use_discriminator = "teacher_response" in data.batch

        if use_discriminator:
            # GAD mode: need both student and teacher data
            select_keys = [
                "input_ids", "responses", "attention_mask", "position_ids",
                "teacher_input_ids", "teacher_response", "teacher_attention_mask", "teacher_position_ids"
            ]
        else:
            # Standard PPO mode: only need student data and returns
            select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    if use_discriminator:
                        # GAD discriminator training
                        responses = model_inputs["responses"]
                        attention_mask = model_inputs["attention_mask"]
                        response_length = responses.size(1)
                        response_mask = attention_mask[:, -response_length:]
                        
                        teacher_response = model_inputs["teacher_response"]
                        teacher_attention_mask = model_inputs["teacher_attention_mask"]
                        teacher_response_length = teacher_response.size(1)
                        teacher_response_mask = teacher_attention_mask[:, -teacher_response_length:]
                        
                        # Randomized dual forward pass to prevent order dependency
                        # Key insight: If critic always sees teacher after student, it may learn
                        # to rely on the order rather than the content quality.
                        # Solution: Randomly shuffle the forward order for each micro-batch.
                        import random
                        if random.random() < 0.5:
                            # Order 1: Teacher first, then student
                            teacher_vpreds = self._forward_micro_batch(model_inputs, compute_teacher=True)
                            student_vpreds = self._forward_micro_batch(model_inputs, compute_teacher=False)
                        else:
                            # Order 2: Student first, then teacher (original order)
                            student_vpreds = self._forward_micro_batch(model_inputs, compute_teacher=False)
                            teacher_vpreds = self._forward_micro_batch(model_inputs, compute_teacher=True)
                        
                        # Compute sequence-level scores for accuracy calculation
                        # Note: vpreds already have last_token_mask applied, so sum gives the last token value
                        teacher_score = teacher_vpreds.sum(dim=-1)  # Last token value (others are 0)
                        student_score = student_vpreds.sum(dim=-1)  # Last token value (others are 0)
                        
                        # Compute discriminator accuracy (per-sample comparison)
                        # For GAD: teacher should score higher than student
                        d_acc = (teacher_score > student_score).float().mean()
                        
                        # Compute discriminator loss (now returns tuple with loss_info)
                        d_loss, loss_info = core_algos.compute_discriminator_loss(
                            student_vpreds=student_vpreds,
                            teacher_vpreds=teacher_vpreds,
                            response_mask=response_mask,
                            teacher_response_mask=teacher_response_mask,
                        )
                        
                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                            loss = d_loss * loss_scale_factor
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation
                            loss = d_loss * loss_scale_factor
                        
                        loss.backward()
                        
                        # 计算长度信息（用于监控）
                        student_lengths = response_mask.sum(dim=-1).float()
                        teacher_lengths = teacher_response_mask.sum(dim=-1).float()
                        
                        micro_batch_metrics.update(
                            {
                                "critic/d_loss": d_loss.detach().item(),
                                "critic/d_acc": d_acc.detach().item(),
                                "critic/student_value_mean": student_score.mean().detach().item(),
                                "critic/teacher_value_mean": teacher_score.mean().detach().item(),
                                "critic/raw_score_diff": (teacher_score - student_score).mean().detach().item(),
                                "critic/ranking_loss": loss_info["ranking_loss"],
                                "critic/score_diff": loss_info["score_diff"],
                                "critic/score_reg": loss_info.get("score_reg", 0.0),
                                "critic/diff_penalty": loss_info.get("diff_penalty", 0.0),
                                "critic/teacher_score_mean": loss_info.get("teacher_score_mean", 0.0),
                                "critic/student_score_mean": loss_info.get("student_score_mean", 0.0),
                                # 简化的长度指标
                                "critic/student_length": student_lengths.mean().detach().item(),
                                "critic/teacher_length": teacher_lengths.mean().detach().item(),
                                # 新增诊断指标
                                "critic/score_diff_abs": torch.abs(teacher_score - student_score).mean().detach().item(),
                                "critic/teacher_score_std": teacher_score.std().detach().item(),
                                "critic/student_score_std": student_score.std().detach().item(),
                                "critic/teacher_score_max": teacher_score.max().detach().item(),
                                "critic/teacher_score_min": teacher_score.min().detach().item(),
                                "critic/student_score_max": student_score.max().detach().item(),
                                "critic/student_score_min": student_score.min().detach().item(),
                                # 分数重叠度：衡量 teacher 和 student 分布的重叠程度
                                "critic/score_overlap": ((teacher_score < student_score.mean()).float().mean() + 
                                                        (student_score > teacher_score.mean()).float().mean()).detach().item() / 2,
                            }
                        )
                    else:
                        # Standard PPO value function training
                        response_mask = model_inputs["response_mask"]
                        values = model_inputs["values"]
                        returns = model_inputs["returns"]

                        vpreds = self._forward_micro_batch(model_inputs, compute_teacher=False)
                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                            loss = vf_loss * loss_scale_factor
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation
                            loss = vf_loss * loss_scale_factor

                        loss.backward()

                        micro_batch_metrics.update(
                            {
                                "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                            }
                        )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.critic_optimizer.zero_grad()
        return metrics
