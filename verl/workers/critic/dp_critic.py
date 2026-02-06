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
    
    def _compute_last_token_mask(self, responses, response_mask, compute_teacher=False):
        """
        计算 last token mask，智能跳过 EOS token
        
        关键修复：如果最后一个 token 是 EOS，使用倒数第二个 token
        这解决了 Student (含EOS) 和 Teacher (不含EOS) 提取不同 token 的问题
        
        Args:
            responses: response token IDs, shape (batch, seq_len)
            response_mask: response attention mask, shape (batch, seq_len)
            compute_teacher: 是否是 teacher response
        
        Returns:
            last_token_mask: bool tensor, shape (batch, seq_len)
        """
        batch_size = response_mask.size(0)
        response_lengths = response_mask.sum(dim=1).long()
        
        # 初始的 last token 索引
        last_token_indices = response_lengths - 1
        
        # 获取最后一个有效 token 的 ID
        batch_indices = torch.arange(batch_size, device=response_mask.device)
        
        # 安全地获取 last token IDs（避免索引越界）
        valid_indices = last_token_indices.clamp(min=0, max=responses.size(1) - 1)
        last_token_ids = responses[batch_indices, valid_indices]
        
        # 获取 EOS token ID
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            eos_token_id = self._tokenizer.eos_token_id
        else:
            # Qwen 系列的 EOS token ID
            eos_token_id = 151645
        
        # 检查是否是 EOS token
        is_eos = (last_token_ids == eos_token_id)
        
        # 统计信息（用于调试，可选）
        if torch.any(is_eos):
            eos_count = is_eos.sum().item()
            # 只在第一次或偶尔打印，避免日志过多
            if not hasattr(self, '_eos_warning_shown'):
                self._eos_warning_shown = True
                logger.info(f"{'Teacher' if compute_teacher else 'Student'} responses: "
                           f"{eos_count}/{batch_size} samples have EOS token at the end")
        
        # 如果最后一个是 EOS，使用倒数第二个 token
        # 确保索引有效（至少为 0）
        adjusted_indices = torch.where(
            is_eos,
            (last_token_indices - 1).clamp(min=0),
            last_token_indices
        )
        
        # 额外检查：如果 response 只有 1 个 token 且是 EOS，使用该 token
        # （虽然这种情况不应该发生，但为了健壮性）
        single_token_eos = (response_lengths == 1) & is_eos
        if torch.any(single_token_eos):
            logger.warning(f"Found {single_token_eos.sum().item()} responses with only EOS token")
            adjusted_indices = torch.where(single_token_eos, last_token_indices, adjusted_indices)
        
        # 创建 mask
        last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
        last_token_mask[batch_indices, adjusted_indices] = True
        
        return last_token_mask

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
                
                # 🔧 修改：使用平均值而非 last token
                # 原因：last token 机制导致 Critic 无法理解语义，只是比较随机的 token value
                # 平均值机制强制模型通过梯度反向传播学习整个序列的语义
                response_mask = attention_mask[:, -response_length:]
                
                # 🔧 关键修复：显式排除 EOS token
                # 问题：response_mask 包含 EOS token，导致相同文本的 Student 和 Teacher 平均值不同
                # 解决：创建排除 EOS 的 mask
                if compute_teacher:
                    response_ids = micro_batch["teacher_response"]
                else:
                    response_ids = micro_batch["responses"]
                
                # 获取 EOS token ID
                if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                    eos_token_id = self._tokenizer.eos_token_id
                else:
                    eos_token_id = 151645  # Qwen 系列默认 EOS token ID
                
                # 找到 EOS token 的位置并排除
                is_eos = (response_ids == eos_token_id)
                response_mask_no_eos = response_mask & (~is_eos)
                
                # 使用排除 EOS 的 mask 计算平均值
                values_sum = (values * response_mask_no_eos).sum(dim=-1)  # (batch,)
                values_count = response_mask_no_eos.sum(dim=-1).clamp(min=1)  # (batch,)
                sequence_value = values_sum / values_count  # (batch,)
                
                # 🔧 修复：确保数据类型一致（BFloat16）
                sequence_value = sequence_value.to(values.dtype)
                
                # 为了保持接口一致（后续代码期望 shape 为 (batch, seq_len)）
                # 将平均值放在最后一个有效位置，其他位置为 0
                values_output = torch.zeros_like(values)
                last_indices = (response_mask_no_eos.sum(dim=-1) - 1).long().clamp(min=0)
                batch_indices = torch.arange(values.size(0), device=values.device)
                values_output[batch_indices, last_indices] = sequence_value
                
                return values_output
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
                
                # 🔧 修改：使用平均值而非 last token
                response_mask = attention_mask[:, -response_length:]
                
                # 🔧 关键修复：显式排除 EOS token
                if compute_teacher:
                    response_ids = micro_batch["teacher_response"]
                else:
                    response_ids = micro_batch["responses"]
                
                # 获取 EOS token ID
                if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                    eos_token_id = self._tokenizer.eos_token_id
                else:
                    eos_token_id = 151645  # Qwen 系列默认 EOS token ID
                
                # 找到 EOS token 的位置并排除
                is_eos = (response_ids == eos_token_id)
                response_mask_no_eos = response_mask & (~is_eos)
                
                # 使用排除 EOS 的 mask 计算平均值
                values_sum = (values * response_mask_no_eos).sum(dim=-1)  # (batch,)
                values_count = response_mask_no_eos.sum(dim=-1).clamp(min=1)  # (batch,)
                sequence_value = values_sum / values_count  # (batch,)
                
                # 🔧 修复：确保数据类型一致（BFloat16）
                sequence_value = sequence_value.to(values.dtype)
                
                # 为了保持接口一致，将平均值放在最后一个有效位置
                values_output = torch.zeros_like(values)
                last_indices = (response_mask_no_eos.sum(dim=-1) - 1).long().clamp(min=0)
                batch_indices = torch.arange(values.size(0), device=values.device)
                values_output[batch_indices, last_indices] = sequence_value
                
                return values_output

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

        # 优化：记录裁剪前的梯度范数
        if isinstance(self.critic_module, FSDP):
            # 计算裁剪前的梯度范数
            grad_norm_before_clip = sum(
                p.grad.data.norm(2).item() ** 2 
                for p in self.critic_module.parameters() 
                if p.grad is not None
            ) ** 0.5
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            # 计算裁剪前的梯度范数
            grad_norm_before_clip = sum(
                p.grad.data.norm(2).item() ** 2 
                for p in self.critic_module.parameters() 
                if p.grad is not None
            ) ** 0.5
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            # 计算裁剪前的梯度范数
            grad_norm_before_clip = sum(
                p.grad.data.norm(2).item() ** 2 
                for p in self.critic_module.parameters() 
                if p.grad is not None
            ) ** 0.5
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
            return grad_norm, grad_norm_before_clip
        else:
            self.critic_optimizer.step()
        return grad_norm, grad_norm_before_clip

    def _log_scoring_details(self, model_inputs, teacher_score, student_score, 
                            response_mask, teacher_response_mask, step):
        """
        记录详细的打分信息，用于人工监控 Critic 的打分是否准确
        
        每 10 步记录一次，显示：
        - 当前 prompt
        - 所有 student responses 及其分数
        - Teacher response 及其分数
        - 分数差异
        
        日志会保存到文件和控制台
        """
        # 只在 rank 0 记录，避免多进程重复
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except:
            pass
        
        # 初始化详细日志记录器（只初始化一次）
        if not hasattr(self, '_detail_logger'):
            import logging
            from datetime import datetime
            
            # 创建专门的详细日志记录器
            self._detail_logger = logging.getLogger('critic_scoring_details')
            self._detail_logger.setLevel(logging.INFO)
            
            # 避免重复添加 handler
            if not self._detail_logger.handlers:
                # 创建日志目录
                log_dir = os.path.join(os.getcwd(), 'logs', 'critic_scoring_details')
                os.makedirs(log_dir, exist_ok=True)
                
                # 创建文件 handler（带时间戳）
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = os.path.join(log_dir, f'scoring_details_{timestamp}.log')
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                
                # 创建控制台 handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                
                # 设置格式（简洁格式，不需要时间戳，因为我们自己会记录 step）
                formatter = logging.Formatter('%(message)s')
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)
                
                self._detail_logger.addHandler(file_handler)
                self._detail_logger.addHandler(console_handler)
                
                # 记录日志文件位置
                logger.info(f"Critic scoring details will be saved to: {log_file}")
        
        try:
            # 获取 tokenizer（如果可用）
            from transformers import AutoTokenizer
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model.path if hasattr(self.config.model, 'path') else 'Qwen/Qwen2.5-7B',
                    trust_remote_code=True
                )
            tokenizer = self._tokenizer
        except Exception as e:
            # 如果无法加载 tokenizer，跳过记录
            logger.warning(f"Failed to load tokenizer for scoring details: {e}")
            return
        
        # 构建完整的输出字符串
        output_lines = []
        output_lines.append("\n" + "="*100)
        output_lines.append(f"📊 Critic 打分详情 - Step {step}")
        output_lines.append("="*100)
        
        try:
            batch_size = teacher_score.size(0)
            
            # 添加批次信息诊断
            output_lines.append(f"\n🔍 批次信息:")
            output_lines.append(f"  总样本数: {batch_size}")
            output_lines.append(f"  Input IDs shape: {model_inputs['input_ids'].shape}")
            output_lines.append(f"  Responses shape: {model_inputs['responses'].shape}")
            output_lines.append(f"  Teacher response shape: {model_inputs['teacher_response'].shape}")
            output_lines.append("")
            
            # 只显示前 2 组样本（每组包含 prompt + student response + teacher response）
            num_samples_to_show = min(2, batch_size)
            
            for sample_idx in range(num_samples_to_show):
                output_lines.append("\n" + "="*100)
                output_lines.append(f"� 样本 #{sample_idx + 1}")
                output_lines.append("="*100)
                
                # 解码 prompt（从 input_ids 中提取，去掉 response 部分）
                input_ids = model_inputs["input_ids"][sample_idx].cpu()
                responses = model_inputs["responses"][sample_idx].cpu()
                attention_mask = model_inputs["attention_mask"][sample_idx].cpu()
                
                # 计算 prompt 长度
                response_length = responses.size(0)
                prompt_length = input_ids.size(0) - response_length
                prompt_ids = input_ids[:prompt_length]
                
                # 解码 prompt（完整显示，不截断）
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                
                output_lines.append(f"\n📝 Prompt:")
                output_lines.append(f"  {prompt_text}")
                output_lines.append("")
                
                # 显示对应的 student response
                output_lines.append(f"🎓 Student Response:")
                output_lines.append("-" * 100)
                
                response_ids = model_inputs["responses"][sample_idx].cpu()
                response_mask_i = response_mask[sample_idx].cpu()
                
                # 只解码有效的 tokens 
                valid_length = response_mask_i.sum().item()
                valid_response_ids = response_ids[:int(valid_length)]
                
                response_text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                score = student_score[sample_idx].item()
                
                output_lines.append(f"  Score: {score:7.4f} | Length: {int(valid_length):3d}")
                output_lines.append(f"  Text: {response_text}")
                output_lines.append(f"  Tokens: {valid_response_ids.tolist()}")
                output_lines.append("")
                
                # 显示对应的 teacher response
                output_lines.append(f"👨‍🏫 Teacher Response:")
                output_lines.append("-" * 100)
                
                teacher_response_ids = model_inputs["teacher_response"][sample_idx].cpu()
                teacher_mask = teacher_response_mask[sample_idx].cpu()
                valid_length = teacher_mask.sum().item()
                valid_teacher_ids = teacher_response_ids[:int(valid_length)]
                
                teacher_text = tokenizer.decode(valid_teacher_ids, skip_special_tokens=True)
                teacher_score_val = teacher_score[sample_idx].item()
                
                output_lines.append(f"  Score: {teacher_score_val:7.4f} | Length: {int(valid_length):3d}")
                output_lines.append(f"  Text: {teacher_text}")
                output_lines.append(f"  Tokens: {valid_teacher_ids.tolist()}")
                output_lines.append("")
                
                # 显示分数对比
                score_diff = teacher_score_val - score
                output_lines.append(f"📊 分数对比:")
                output_lines.append(f"  Teacher - Student = {score_diff:7.4f}")
                output_lines.append(f"  Teacher > Student: {'✅ 正确' if teacher_score_val > score else '❌ 错误' if teacher_score_val < score else '⚖️  相等'}")
                
                # 检查内容相似度（简单的文本匹配）
                if teacher_text.strip() == response_text.strip():
                    # 检查 token 长度是否相同
                    student_token_len = len(valid_response_ids.tolist())
                    teacher_token_len = len(valid_teacher_ids.tolist())
                    
                    # 只有当分数差异显著时才警告
                    if abs(score_diff) > 0.5:
                        output_lines.append(f"  ⚠️  警告: Teacher 和 Student 回答完全相同，但分数差异为 {abs(score_diff):.4f}!")
                        
                        if student_token_len != teacher_token_len:
                            output_lines.append(f"  🚨 关键发现: 相同文本但 token 长度不同!")
                            output_lines.append(f"     Student tokens: {student_token_len}")
                            output_lines.append(f"     Teacher tokens: {teacher_token_len}")
                            output_lines.append(f"     这可能是分数差异的根本原因！")
                    elif student_token_len != teacher_token_len:
                        # 分数相同但长度不同，说明修复生效
                        output_lines.append(f"  ✅ 相同文本，分数一致 (分差: {abs(score_diff):.4f})")
                        output_lines.append(f"  📝 注: Student 包含 EOS token ({student_token_len} tokens)，Teacher 不包含 ({teacher_token_len} tokens)")
                        output_lines.append(f"     EOS token 已被正确跳过，提取了相同位置的 token")
                    else:
                        # 完美情况：长度和分数都相同
                        output_lines.append(f"  ✅ 完美: 相同文本，相同长度，相同分数")
            
            output_lines.append("\n" + "="*100)
            
            # 全局统计信息
            output_lines.append(f"\n� 全局统计信息 (共 {batch_size} 个样本):")
            output_lines.append("-" * 100)
            output_lines.append(f"  Teacher 平均分: {teacher_score.mean().item():7.4f}")
            output_lines.append(f"  Student 平均分: {student_score.mean().item():7.4f}")
            output_lines.append(f"  平均分差:       {(teacher_score - student_score).mean().item():7.4f}")
            output_lines.append(f"  Teacher > Student: {(teacher_score > student_score).float().mean().item()*100:.1f}%")
            
            # 显示分数分布
            output_lines.append(f"\n  Student 分数范围: [{student_score.min().item():.4f}, {student_score.max().item():.4f}]")
            output_lines.append(f"  Teacher 分数范围: [{teacher_score.min().item():.4f}, {teacher_score.max().item():.4f}]")
            
            # 检查相同答案的分数差异（诊断顺序依赖问题）
            same_answer_count = 0
            same_answer_score_diffs = []
            for i in range(batch_size):
                try:
                    student_text = tokenizer.decode(
                        model_inputs["responses"][i].cpu()[:int(response_mask[i].sum().item())],
                        skip_special_tokens=True
                    ).strip()
                    teacher_text = tokenizer.decode(
                        model_inputs["teacher_response"][i].cpu()[:int(teacher_response_mask[i].sum().item())],
                        skip_special_tokens=True
                    ).strip()
                    
                    if student_text == teacher_text and len(student_text) > 0:
                        same_answer_count += 1
                        score_diff = abs(teacher_score[i].item() - student_score[i].item())
                        same_answer_score_diffs.append(score_diff)
                except:
                    pass
            
            if same_answer_count > 0:
                avg_diff = sum(same_answer_score_diffs) / len(same_answer_score_diffs)
                output_lines.append(f"\n⚠️  顺序依赖诊断:")
                output_lines.append(f"  相同答案数量: {same_answer_count}/{batch_size}")
                output_lines.append(f"  相同答案的平均分差: {avg_diff:.4f}")
                
                # 更新警告阈值：修复后应该 < 0.5
                if avg_diff > 1.0:
                    output_lines.append(f"  🚨 警告: 相同答案分差过大 (>{avg_diff:.2f})，可能存在严重的顺序依赖问题!")
                elif avg_diff > 0.5:
                    output_lines.append(f"  ⚠️  注意: 相同答案分差略高 ({avg_diff:.2f})，建议继续观察")
                else:
                    output_lines.append(f"  ✅ 良好: 相同答案分差很小 ({avg_diff:.2f})，EOS token 修复生效!")
            
        except Exception as e:
            output_lines.append(f"  ⚠️  记录详情时出错: {e}")
            import traceback
            output_lines.append(f"  错误堆栈: {traceback.format_exc()}")
            logger.warning(f"Error in _log_scoring_details: {e}")
        
        output_lines.append("\n" + "="*100 + "\n")
        
        # 使用 logger 记录（会同时输出到文件和控制台）
        self._detail_logger.info("\n".join(output_lines))

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        # 优化：记录裁剪前的梯度范数
        if isinstance(self.critic_module, FSDP):
            # 计算裁剪前的梯度范数
            grad_norm_before_clip = sum(
                p.grad.data.norm(2).item() ** 2 
                for p in self.critic_module.parameters() 
                if p.grad is not None
            ) ** 0.5
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            # 计算裁剪前的梯度范数
            grad_norm_before_clip = sum(
                p.grad.data.norm(2).item() ** 2 
                for p in self.critic_module.parameters() 
                if p.grad is not None
            ) ** 0.5
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            # 计算裁剪前的梯度范数
            grad_norm_before_clip = sum(
                p.grad.data.norm(2).item() ** 2 
                for p in self.critic_module.parameters() 
                if p.grad is not None
            ) ** 0.5
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
            return grad_norm, grad_norm_before_clip
        else:
            self.critic_optimizer.step()
        return grad_norm, grad_norm_before_clip

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
        
        # 添加全局步数计数器（用于控制打印频率）
        if not hasattr(self, '_update_step'):
            self._update_step = 0
        self._update_step += 1
        
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
                        
                        # 每 5 步打印详细的打分信息（用于人工监控）
                        if self._update_step % 5 == 0 and batch_idx == 0:
                            self._log_scoring_details(
                                model_inputs=model_inputs,
                                teacher_score=teacher_score,
                                student_score=student_score,
                                response_mask=response_mask,
                                teacher_response_mask=teacher_response_mask,
                                step=self._update_step
                            )
                        
                        # Compute discriminator accuracy (per-sample comparison)
                        # 使用软标签方案：将分数差异映射到 [0, 1] 区间
                        # 优势：
                        # 1. score_diff = 0 → d_acc = 0.5（中性，相同质量）
                        # 2. score_diff > 0 → d_acc > 0.5（teacher 更好）
                        # 3. score_diff < 0 → d_acc < 0.5（student 更好，异常）
                        # 4. 连续可微，避免硬阈值导致的不连续性
                        score_diff = teacher_score - student_score
                        d_acc = torch.sigmoid(score_diff).mean()  # 软标签
                        
                        # 同时保留硬标签用于对比（仅用于监控）
                        d_acc_hard = (teacher_score > student_score).float().mean()
                        
                        # Compute discriminator loss (now returns tuple with loss_info)
                        # 优化 2026-01-29：
                        # 1. 增大 temperature 从 0.5 到 5.0，缓解梯度饱和
                        # 2. 关闭自适应 temperature，使用固定值
                        # 3. 移除一致性损失（EOS Token 问题已通过显式排除修复）
                        # 4. 移除混合分数和 Batch Normalization（简化代码，直接使用原始分数）
                        d_loss, loss_info = core_algos.compute_discriminator_loss(
                            student_vpreds=student_vpreds,
                            teacher_vpreds=teacher_vpreds,
                            response_mask=response_mask,
                            teacher_response_mask=teacher_response_mask,
                            temperature=1,  # 从 0.5 增大到 5.0
                            adaptive_temperature=True,  # 关闭自适应，使用固定值
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
                        
                        # 优化后的指标（精简版）：只保留核心指标
                        # 从 core_algos 获取的指标已经简化，这里只添加必要的额外指标
                        micro_batch_metrics.update(
                            {
                                # 核心损失和准确率（3 个）
                                "critic/d_loss": d_loss.detach().item(),
                                "critic/d_acc": d_acc.detach().item(),  # 软标签准确率
                                
                                # 从 core_algos 获取的核心指标（6 个）
                                "critic/ranking_loss": loss_info["ranking_loss"],
                                "critic/score_reg": loss_info["score_reg"],
                                "critic/score_diff": loss_info["score_diff"],
                                "critic/teacher_score_mean": loss_info["teacher_score_mean"],
                                "critic/student_score_mean": loss_info["student_score_mean"],
                                "critic/temperature": loss_info["temperature"],
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

                grad_norm, grad_norm_before_clip = self._optimizer_step()
                # 梯度指标（1 个）- 只保留最关键的梯度范数
                mini_batch_metrics = {
                    "critic/grad_norm": grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm,
                }
                append_to_dict(metrics, mini_batch_metrics)
        self.critic_optimizer.zero_grad()
        return metrics
