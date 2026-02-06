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
Critic 模型训练中评估器

功能：
1. 在训练过程中定期评估 Critic 模型的打分能力
2. 支持直接使用 FSDP 切片模型（无需合并）
3. 生成详细的评估报告和可视化
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CriticEvaluator:
    """
    Critic 模型评估器
    
    在训练过程中定期评估 Critic 模型的打分准确性
    支持直接使用训练中的 Actor 模型（FSDP 切片）生成 responses
    """
    
    def __init__(
        self,
        config: dict,
        critic_module: torch.nn.Module,
        actor_module: Optional[torch.nn.Module] = None,
        tokenizer = None,
        eval_data_path: Optional[str] = None,
        eval_freq: int = 100,
        num_eval_samples: int = 100,
        n_resp_per_prompt: int = 4,
        batch_size: int = 8,
        output_dir: Optional[str] = None,
        generation_config: Optional[dict] = None,
    ):
        """
        Args:
            config: 训练配置
            critic_module: Critic 模型（可以是 FSDP 模型）
            actor_module: Actor 模型（可以是 FSDP 模型，用于生成 student responses）
            tokenizer: Tokenizer
            eval_data_path: 评估数据集路径
            eval_freq: 评估频率（每 N 步评估一次）
            num_eval_samples: 每次评估的样本数
            n_resp_per_prompt: 每个 prompt 生成的 student responses 数量
            batch_size: 批处理大小
            output_dir: 评估结果输出目录
            generation_config: 生成配置（temperature, top_p, max_new_tokens 等）
        """
        self.config = config
        self.critic_module = critic_module
        self.actor_module = actor_module
        self.tokenizer = tokenizer
        self.eval_data_path = eval_data_path
        self.eval_freq = eval_freq
        self.num_eval_samples = num_eval_samples
        self.n_resp_per_prompt = n_resp_per_prompt
        self.batch_size = batch_size
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'critic_eval_results')
        
        # 生成配置
        self.generation_config = generation_config or {
            'temperature': 0.6,
            'top_p': 0.9,
            'max_new_tokens': 512,
            'do_sample': True,
            'repetition_penalty': 1.2,
        }
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 评估历史记录
        self.eval_history = []
        
        # 加载评估数据
        self._load_eval_data()
        
        logger.info(f"CriticEvaluator initialized:")
        logger.info(f"  - Eval frequency: every {eval_freq} steps")
        logger.info(f"  - Eval samples: {num_eval_samples}")
        logger.info(f"  - Responses per prompt: {n_resp_per_prompt}")
        logger.info(f"  - Use Actor model: {actor_module is not None}")
        logger.info(f"  - Generation config: {self.generation_config}")
        logger.info(f"  - Output dir: {self.output_dir}")
    
    def _load_eval_data(self):
        """加载评估数据集"""
        if self.eval_data_path is None:
            logger.warning("No eval_data_path provided, evaluation will be skipped")
            self.eval_data = None
            return
        
        try:
            df = pd.read_parquet(self.eval_data_path)
            
            # 采样
            if len(df) > self.num_eval_samples:
                df = df.sample(n=self.num_eval_samples, random_state=42)
            
            self.eval_data = df
            logger.info(f"Loaded {len(df)} evaluation samples from {self.eval_data_path}")
        except Exception as e:
            logger.error(f"Failed to load eval data: {e}")
            self.eval_data = None
    
    def should_evaluate(self, step: int) -> bool:
        """判断是否应该在当前步数进行评估"""
        return (
            self.eval_data is not None 
            and self.eval_freq > 0 
            and step > 0 
            and step % self.eval_freq == 0
        )
    
    def _generate_student_responses(
        self, 
        prompts: List[str], 
        n_responses: int = 1
    ) -> List[List[str]]:
        """
        使用 Actor 模型生成 student responses
        
        Args:
            prompts: prompt 列表
            n_responses: 每个 prompt 生成的 responses 数量
        
        Returns:
            List[List[str]]: 每个 prompt 对应的 responses 列表
        """
        if self.actor_module is None:
            logger.warning("No Actor module available, using empty responses")
            return [[""] * n_responses for _ in prompts]
        
        try:
            # 获取设备
            if hasattr(self.actor_module, 'pretrained_model'):
                if hasattr(self.actor_module.pretrained_model, 'hf_device_map'):
                    device = list(self.actor_module.pretrained_model.hf_device_map.values())[0]
                else:
                    device = next(self.actor_module.pretrained_model.parameters()).device
            else:
                device = next(self.actor_module.parameters()).device
            
            all_responses = []
            
            # 批量生成（每个 prompt 生成 n_responses 次）
            for prompt in tqdm(prompts, desc="Generating student responses"):
                prompt_responses = []
                
                for _ in range(n_responses):
                    # 准备输入
                    messages = [{"role": "user", "content": prompt}]
                    input_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    inputs = self.tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        truncation=True,
                        max_length=4096
                    )
                    
                    # 移动到设备
                    if not hasattr(self.actor_module, 'pretrained_model') or \
                       not hasattr(self.actor_module.pretrained_model, 'hf_device_map'):
                        inputs = inputs.to(device)
                    else:
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                    
                    # 生成
                    with torch.no_grad():
                        # 从 pretrained_model 获取生成方法（支持 FSDP）
                        if hasattr(self.actor_module, 'pretrained_model'):
                            model_for_generation = self.actor_module.pretrained_model
                        else:
                            model_for_generation = self.actor_module
                        
                        outputs = model_for_generation.generate(
                            **inputs,
                            max_new_tokens=self.generation_config.get('max_new_tokens', 512),
                            temperature=self.generation_config.get('temperature', 0.6),
                            top_p=self.generation_config.get('top_p', 0.9),
                            do_sample=self.generation_config.get('do_sample', True),
                            repetition_penalty=self.generation_config.get('repetition_penalty', 1.2),
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    # 解码（只取新生成的部分）
                    input_length = inputs['input_ids'].shape[1]
                    generated_ids = outputs[0, input_length:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    prompt_responses.append(response.strip())
                
                all_responses.append(prompt_responses)
            
            return all_responses
        
        except Exception as e:
            logger.error(f"Failed to generate student responses: {e}")
            import traceback
            traceback.print_exc()
            return [[""] * n_responses for _ in prompts]
    
    def _get_critic_scores_batch(
        self, 
        prompts: List[str], 
        responses: List[str], 
        max_length: int = 2048
    ) -> Tuple[List[float], List[int]]:
        """
        批量获取 Critic 分数
        
        关键：支持直接使用 FSDP 模型，无需合并
        """
        try:
            # 获取设备
            if hasattr(self.critic_module, 'pretrained_model'):
                if hasattr(self.critic_module.pretrained_model, 'hf_device_map'):
                    device = list(self.critic_module.pretrained_model.hf_device_map.values())[0]
                else:
                    device = next(self.critic_module.pretrained_model.parameters()).device
            else:
                device = next(self.critic_module.parameters()).device
            
            batch_size = len(prompts)
            
            # 准备 batch 数据
            all_input_texts = []
            for prompt, response in zip(prompts, responses):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                all_input_texts.append(input_text)
            
            # Batch tokenization
            inputs = self.tokenizer(
                all_input_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            
            # 计算 response 长度
            all_response_lengths = []
            for prompt, response in zip(prompts, responses):
                prompt_messages = [{"role": "user", "content": prompt}]
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)
                prompt_length = len(prompt_tokens['input_ids'])
                
                full_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                full_text = self.tokenizer.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
                full_tokens = self.tokenizer(full_text, add_special_tokens=False)
                full_length = len(full_tokens['input_ids'])
                
                response_length = full_length - prompt_length
                all_response_lengths.append(response_length)
            
            # 移动到设备
            if not hasattr(self.critic_module, 'pretrained_model') or \
               not hasattr(self.critic_module.pretrained_model, 'hf_device_map'):
                inputs = inputs.to(device)
            else:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.critic_module(**inputs, use_cache=False)
                
                # 提取 values
                if hasattr(self.critic_module, "v_head"):
                    all_values = outputs[2]
                    if all_values.dim() == 3:
                        all_values = all_values.squeeze(-1)
                else:
                    all_values = outputs.logits
                    if all_values.dim() == 3:
                        all_values = all_values.squeeze(-1)
                
                # 提取每个样本的分数
                scores = []
                lengths = []
                
                for i in range(batch_size):
                    response_length = all_response_lengths[i]
                    values = all_values[i:i+1, -response_length:]
                    
                    # 获取 mask 并排除 EOS token
                    attention_mask = inputs['attention_mask'][i:i+1]
                    response_mask = attention_mask[:, -response_length:]
                    response_ids = inputs['input_ids'][i:i+1, -response_length:]
                    
                    eos_token_id = self.tokenizer.eos_token_id
                    is_eos = (response_ids == eos_token_id)
                    response_mask_no_eos = response_mask & (~is_eos)
                    
                    # 计算平均分数（与训练时一致）
                    values_sum = (values * response_mask_no_eos).sum(dim=-1)
                    length = response_mask_no_eos.sum(dim=-1).clamp(min=1)
                    score_avg = (values_sum / length).item()
                    
                    scores.append(score_avg)
                    lengths.append(length.item())
                
                return scores, lengths
        
        except Exception as e:
            logger.error(f"Batch Critic scoring failed: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * len(prompts), [0] * len(prompts)
    
    def evaluate(self, step: int) -> Dict[str, float]:
        """
        执行评估
        
        Args:
            step: 当前训练步数
        
        Returns:
            评估指标字典
        """
        if self.eval_data is None:
            logger.warning("No eval data available, skipping evaluation")
            return {}
        
        logger.info(f"Starting Critic evaluation at step {step}...")
        start_time = time.time()
        
        # 切换到评估模式
        was_training = self.critic_module.training
        self.critic_module.eval()
        
        try:
            # 准备数据
            all_prompts = []
            all_teacher_responses = []
            
            for idx, row in self.eval_data.iterrows():
                try:
                    content = row['content']
                    if isinstance(content, (list, tuple)) and len(content) > 0:
                        prompt = content[0].get('content', '') if isinstance(content[0], dict) else str(content[0])
                    else:
                        prompt = str(content)
                    
                    teacher_response = row.get('teacher_response', '')
                    
                    if not prompt or not teacher_response:
                        continue
                    
                    all_prompts.append(prompt)
                    all_teacher_responses.append(teacher_response)
                except Exception as e:
                    logger.warning(f"Failed to parse sample: {e}")
                    continue
            
            logger.info(f"Prepared {len(all_prompts)} valid samples")
            
            # 生成 Student responses（使用训练中的 Actor 模型）
            logger.info("Generating Student responses using Actor model...")
            all_student_responses = self._generate_student_responses(
                all_prompts, 
                n_responses=self.n_resp_per_prompt
            )
            
            # Batch 推理
            logger.info("Running batch inference...")
            results = []
            total_correct = 0
            total_comparisons = 0
            
            all_teacher_scores = []
            all_student_scores = []
            
            for batch_start in tqdm(range(0, len(all_prompts), self.batch_size), desc="Evaluating"):
                batch_end = min(batch_start + self.batch_size, len(all_prompts))
                
                batch_prompts = all_prompts[batch_start:batch_end]
                batch_teacher_responses = all_teacher_responses[batch_start:batch_end]
                batch_student_responses = all_student_responses[batch_start:batch_end]
                
                # 构建混合 batch（与训练时一致）
                mixed_prompts = []
                mixed_responses = []
                teacher_indices = []
                student_indices_map = {}
                
                current_idx = 0
                
                # 先添加所有 teachers
                for i, (prompt, teacher_resp) in enumerate(zip(batch_prompts, batch_teacher_responses)):
                    mixed_prompts.append(prompt)
                    mixed_responses.append(teacher_resp)
                    teacher_indices.append(current_idx)
                    current_idx += 1
                
                # 再添加所有 students
                for i, (prompt, student_resps) in enumerate(zip(batch_prompts, batch_student_responses)):
                    student_start_idx = current_idx
                    for student_resp in student_resps:
                        mixed_prompts.append(prompt)
                        mixed_responses.append(student_resp)
                        current_idx += 1
                    student_indices_map[i] = list(range(student_start_idx, current_idx))
                
                # Batch 推理
                all_scores, all_lengths = self._get_critic_scores_batch(
                    mixed_prompts, mixed_responses
                )
                
                # 分离 teacher 和 student 分数
                for i in range(len(batch_prompts)):
                    teacher_score = all_scores[teacher_indices[i]]
                    student_score_indices = student_indices_map[i]
                    student_scores = [all_scores[idx] for idx in student_score_indices]
                    
                    # 统计
                    all_teacher_scores.append(teacher_score)
                    all_student_scores.extend(student_scores)
                    
                    # 计算准确率
                    correct = sum(1 for s in student_scores if s <= teacher_score)
                    total_correct += correct
                    total_comparisons += len(student_scores)
                    
                    results.append({
                        'teacher_score': teacher_score,
                        'student_scores': student_scores,
                        'correct': correct,
                        'total': len(student_scores),
                    })
            
            # 计算评估指标
            accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0.0
            teacher_mean = np.mean(all_teacher_scores)
            student_mean = np.mean(all_student_scores)
            teacher_std = np.std(all_teacher_scores)
            student_std = np.std(all_student_scores)
            score_diff = teacher_mean - student_mean
            
            metrics = {
                'eval/accuracy': accuracy,
                'eval/teacher_score_mean': teacher_mean,
                'eval/student_score_mean': student_mean,
                'eval/teacher_score_std': teacher_std,
                'eval/student_score_std': student_std,
                'eval/score_diff': score_diff,
                'eval/num_samples': len(results),
                'eval/num_comparisons': total_comparisons,
            }
            
            # 记录评估历史
            eval_record = {
                'step': step,
                'timestamp': time.time(),
                **metrics
            }
            self.eval_history.append(eval_record)
            
            # 保存详细结果
            self._save_eval_results(step, results, metrics)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Evaluation completed in {elapsed_time:.2f}s")
            logger.info(f"  Accuracy: {accuracy*100:.2f}%")
            logger.info(f"  Teacher score: {teacher_mean:.4f} ± {teacher_std:.4f}")
            logger.info(f"  Student score: {student_mean:.4f} ± {student_std:.4f}")
            logger.info(f"  Score diff: {score_diff:.4f}")
            
            return metrics
        
        finally:
            # 恢复训练模式
            if was_training:
                self.critic_module.train()
    
    def _save_eval_results(self, step: int, results: List[dict], metrics: Dict[str, float]):
        """保存评估结果"""
        try:
            # 保存详细结果
            results_file = os.path.join(self.output_dir, f'eval_step_{step}_results.json')
            import json
            with open(results_file, 'w') as f:
                json.dump({
                    'step': step,
                    'metrics': metrics,
                    'results': results
                }, f, indent=2)
            
            # 保存评估历史
            history_file = os.path.join(self.output_dir, 'eval_history.csv')
            df = pd.DataFrame(self.eval_history)
            df.to_csv(history_file, index=False)
            
            logger.info(f"Eval results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save eval results: {e}")
