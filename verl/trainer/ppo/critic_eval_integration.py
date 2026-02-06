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
Critic 评估集成到训练流程的辅助函数

使用方法：
1. 在训练脚本中导入 setup_critic_evaluator
2. 在训练循环中调用 maybe_evaluate_critic
"""

import logging
from typing import Optional

from verl.trainer.ppo.critic_evaluator import CriticEvaluator

logger = logging.getLogger(__name__)


def setup_critic_evaluator(
    config,
    critic_module,
    actor_module,
    tokenizer,
) -> Optional[CriticEvaluator]:
    """
    根据配置创建 Critic 评估器
    
    Args:
        config: 训练配置（Hydra/OmegaConf）
        critic_module: Critic 模型
        actor_module: Actor 模型
        tokenizer: Tokenizer
    
    Returns:
        CriticEvaluator 实例，如果未启用则返回 None
    """
    # 检查是否启用评估
    if not config.get('critic_evaluation', {}).get('enable', False):
        logger.info("Critic evaluation is disabled")
        return None
    
    eval_config = config.critic_evaluation
    
    # 创建评估器
    evaluator = CriticEvaluator(
        config=config,
        critic_module=critic_module,
        actor_module=actor_module,
        tokenizer=tokenizer,
        eval_data_path=eval_config.get('eval_data_path'),
        eval_freq=eval_config.get('eval_freq', 100),
        num_eval_samples=eval_config.get('num_eval_samples', 100),
        n_resp_per_prompt=eval_config.get('n_resp_per_prompt', 4),
        batch_size=eval_config.get('batch_size', 8),
        output_dir=eval_config.get('output_dir'),
        generation_config=eval_config.get('generation_config', {
            'temperature': 0.6,
            'top_p': 0.9,
            'max_new_tokens': 512,
            'do_sample': True,
            'repetition_penalty': 1.2,
        }),
    )
    
    logger.info("Critic evaluator created successfully")
    return evaluator


def maybe_evaluate_critic(
    evaluator: Optional[CriticEvaluator],
    step: int,
    logger_obj=None,
) -> dict:
    """
    如果满足条件，执行 Critic 评估
    
    Args:
        evaluator: CriticEvaluator 实例
        step: 当前训练步数
        logger_obj: 日志记录器（用于记录到 TensorBoard/WandB）
    
    Returns:
        评估指标字典，如果未评估则返回空字典
    """
    if evaluator is None:
        return {}
    
    if not evaluator.should_evaluate(step):
        return {}
    
    try:
        # 执行评估
        metrics = evaluator.evaluate(step)
        
        # 记录到日志系统
        if logger_obj is not None and metrics:
            logger_obj.log(data=metrics, step=step)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Critic evaluation failed at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================
# 使用示例（在 ray_trainer.py 的 fit 方法中）
# ============================================================
"""
在 ray_trainer.py 的 __init__ 方法中添加：

    from verl.trainer.ppo.critic_eval_integration import setup_critic_evaluator
    
    # 在初始化 critic 之后
    self.critic_evaluator = setup_critic_evaluator(
        config=self.config,
        critic_module=self.critic_wg.critic_module,  # 需要暴露 critic_module
        actor_module=self.actor_rollout_wg.actor_module,  # 需要暴露 actor_module
        tokenizer=self.tokenizer,
    )


在 ray_trainer.py 的 fit 方法的训练循环中添加：

    from verl.trainer.ppo.critic_eval_integration import maybe_evaluate_critic
    
    # 在每个训练步结束后
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # ... 训练代码 ...
            
            # 记录指标
            logger.log(data=metrics, step=self.global_steps)
            
            # 🔧 新增：Critic 评估
            eval_metrics = maybe_evaluate_critic(
                evaluator=self.critic_evaluator,
                step=self.global_steps,
                logger_obj=logger,
            )
            
            # 更新进度条
            progress_bar.update(1)
            self.global_steps += 1
            
            # ... 其他代码 ...
"""
