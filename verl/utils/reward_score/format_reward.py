# Copyright 2024 Custom Implementation
# Format Reward for GAD Training
# 用于惩罚格式不遵循的行为

import re
import json
from typing import Optional, Dict, Any, Union


def compute_format_reward(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    计算格式奖励，用于惩罚以下问题：
    1. 时间戳泄漏：输出中包含 [YYYY-MM-DD HH:MM:SS] 格式
    2. JSON 格式不完整：缺少闭合括号
    3. 重复输出：同一短语重复多次
    4. 输出过长：超过合理长度
    
    Args:
        solution_str: 模型生成的输出
        ground_truth: 期望的输出格式（用于参考）
        extra_info: 额外信息
        
    Returns:
        float: 格式奖励分数 [-1.0, 1.0]
        或 dict: 包含详细信息的字典
    """
    reward = 0.0
    penalties = {}
    
    # ========== 1. 时间戳泄漏检测 ==========
    # 匹配 [2025-12-16 10:12:00] 格式的时间戳
    timestamp_pattern = r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]'
    timestamps_found = re.findall(timestamp_pattern, solution_str)
    if timestamps_found:
        # 每个时间戳扣 0.3 分
        timestamp_penalty = min(len(timestamps_found) * 0.3, 0.9)
        reward -= timestamp_penalty
        penalties["timestamp_leak"] = {
            "count": len(timestamps_found),
            "penalty": timestamp_penalty,
            "examples": timestamps_found[:3]  # 最多记录3个
        }
    
    # ========== 2. JSON 格式检测 ==========
    # 检测是否应该是 JSON 格式
    json_start_pattern = r'^\s*\{|"conclusion"|"rewritten_query"|"analysis"'
    if re.search(json_start_pattern, solution_str):
        # 尝试解析 JSON
        try:
            # 尝试提取 JSON 部分
            json_match = re.search(r'\{[^{}]*\}', solution_str, re.DOTALL)
            if json_match:
                json.loads(json_match.group())
                # JSON 格式正确，给予小奖励
                reward += 0.1
            else:
                # 有 JSON 开头但没有完整结构
                reward -= 0.5
                penalties["json_incomplete"] = {
                    "reason": "JSON structure not found",
                    "penalty": 0.5
                }
        except json.JSONDecodeError as e:
            # JSON 解析失败
            reward -= 0.4
            penalties["json_invalid"] = {
                "reason": str(e),
                "penalty": 0.4
            }
    
    # ========== 3. 重复输出检测 ==========
    # 检测连续重复的短语（至少10个字符重复3次以上）
    repetition_pattern = r'(.{10,}?)\1{2,}'
    repetitions = re.findall(repetition_pattern, solution_str)
    if repetitions:
        # 重复是严重问题，重罚
        repetition_penalty = min(len(repetitions) * 0.4, 1.0)
        reward -= repetition_penalty
        penalties["repetition"] = {
            "count": len(repetitions),
            "penalty": repetition_penalty,
            "examples": [r[:50] + "..." if len(r) > 50 else r for r in repetitions[:2]]
        }
    
    # ========== 4. 输出长度检测 ==========
    # 如果有 ground_truth，比较长度
    if ground_truth:
        gt_len = len(ground_truth)
        sol_len = len(solution_str)
        
        # 如果输出长度超过参考的 2 倍，惩罚
        if gt_len > 0 and sol_len > gt_len * 2:
            length_ratio = sol_len / gt_len
            length_penalty = min((length_ratio - 2) * 0.1, 0.3)
            reward -= length_penalty
            penalties["too_long"] = {
                "solution_len": sol_len,
                "ground_truth_len": gt_len,
                "ratio": round(length_ratio, 2),
                "penalty": length_penalty
            }
    
    # ========== 5. 特殊字符污染检测 ==========
    # 检测不应该出现的特殊字符
    special_chars = ['\t', '\r', '\x00']
    for char in special_chars:
        if char in solution_str:
            reward -= 0.1
            penalties["special_char"] = {
                "char": repr(char),
                "penalty": 0.1
            }
            break
    
    # 限制奖励范围
    reward = max(min(reward, 1.0), -1.0)
    
    return {
        "score": reward,
        "penalties": penalties,
        "solution_preview": solution_str[:200] + "..." if len(solution_str) > 200 else solution_str
    }


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    """
    简化接口，只返回分数
    """
    result = compute_format_reward(solution_str, ground_truth, extra_info, **kwargs)
    if isinstance(result, dict):
        return result["score"]
    return result


# 用于测试
if __name__ == "__main__":
    # 测试用例 1: 时间戳泄漏
    test1 = '{"conclusion": "否", "analysis": "销售前后矛盾[2025-12-16 10:12:00]'
    result1 = compute_format_reward(test1, "")
    print(f"Test 1 (timestamp): {result1}")
    
    # 测试用例 2: 重复输出
    test2 = '{"rewritten_query": "问题？问题？问题？问题？问题？问题？"}'
    result2 = compute_format_reward(test2, "")
    print(f"Test 2 (repetition): {result2}")
    
    # 测试用例 3: JSON 不完整
    test3 = '{"conclusion": "否", "analysis": "分析内容'
    result3 = compute_format_reward(test3, "")
    print(f"Test 3 (incomplete JSON): {result3}")
    
    # 测试用例 4: 正常输出
    test4 = '{"conclusion": "是", "analysis": "销售回复正确"}'
    result4 = compute_format_reward(test4, "")
    print(f"Test 4 (normal): {result4}")
