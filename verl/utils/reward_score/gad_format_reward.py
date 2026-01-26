# Copyright 2024 Custom Implementation
# GAD Format Reward - 通用版 v3
# 核心设计：以 ground_truth 为参照，检测格式一致性

import re
import json
from typing import Optional, Dict, Any, Tuple


def extract_json_from_text(text: str) -> Optional[str]:
    """从文本中提取第一个完整的 JSON 对象"""
    start = text.find('{')
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_safe(text: str) -> Tuple[Optional[dict], str]:
    """
    安全解析 JSON，返回 (解析结果, 错误类型)
    错误类型: "ok", "missing", "incomplete", "invalid", "prefix"
    """
    if not text or len(text.strip()) < 3:
        return None, "empty"
    
    stripped = text.strip()
    has_brace = '{' in text
    
    # 1. 完全没有 JSON 结构
    if not has_brace:
        return None, "missing"
    
    # 2. 检查前缀污染
    json_start = stripped.find('{')
    if json_start > 0:
        prefix = stripped[:json_start].strip()
        if prefix and len(prefix) > 5:  # 超过 5 字符的前缀
            return None, "prefix"
    
    # 3. 检查括号是否完整
    if text.count('{') > text.count('}'):
        return None, "incomplete"
    
    # 4. 尝试解析 JSON
    json_str = extract_json_from_text(text)
    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed, "ok"
            return None, "invalid"
        except json.JSONDecodeError:
            return None, "invalid"
    
    return None, "invalid"


def check_json_consistency(solution: str, ground_truth: str) -> Tuple[Optional[Dict], Optional[dict]]:
    """
    检测 JSON 格式一致性（通用版）
    
    核心逻辑：
    1. 如果 ground_truth 是有效 JSON，则 solution 也必须是有效 JSON
    2. 如果 ground_truth 包含特定字段，检查 solution 是否也包含
    3. 如果 ground_truth 是纯文本（非 JSON），则不检查 JSON 格式
    
    返回：(惩罚信息, 解析后的JSON)
    """
    # 解析 ground_truth
    gt_json, gt_status = parse_json_safe(ground_truth)
    # 只有当 ground_truth 是有效 JSON 时，才要求 solution 也是 JSON
    # 避免纯文本中恰好包含 '{' 字符导致的误判
    gt_expects_json = gt_status == "ok"
    
    # 解析 solution
    sol_json, sol_status = parse_json_safe(solution)
    
    # 如果 ground_truth 期望 JSON 输出
    if gt_expects_json:
        if sol_status == "missing":
            return {"type": "json_missing", "penalty": 0.5}, None
        elif sol_status == "incomplete":
            return {"type": "json_incomplete", "penalty": 0.3}, None
        elif sol_status == "invalid":
            return {"type": "json_invalid", "penalty": 0.25}, None
        elif sol_status == "prefix":
            return {"type": "json_prefix", "penalty": 0.3}, None
        elif sol_status == "ok":
            # JSON 解析成功，检查字段一致性
            if gt_json and sol_json:
                gt_keys = set(gt_json.keys())
                sol_keys = set(sol_json.keys())
                missing_keys = gt_keys - sol_keys
                if missing_keys:
                    return {"type": "json_keys_missing", "keys": list(missing_keys), "penalty": 0.2}, sol_json
            return None, sol_json  # JSON 正确
    
    # ground_truth 不是 JSON，不检查 JSON 格式
    return None, sol_json


def check_language_pollution(text: str, parsed_json: Optional[dict] = None) -> Optional[Dict]:
    """检测语言污染问题（思考泄露 + 中英混杂）"""
    # 1. 英文思考泄露
    thinking_patterns = r'(?:here is|based on|according to|the (?:provided|given|above)|let me|I will|output (?:is|as)|following (?:is|are)|as (?:requested|required))'
    if re.search(thinking_patterns, text, re.IGNORECASE):
        return {"type": "thinking_leak", "penalty": 0.4}
    
    # 2. 中英文混杂（中文后跟多个英文单词）
    if re.search(r'[\u4e00-\u9fff]\s*[a-zA-Z]{2,}(?:\s+[a-zA-Z]+)+', text):
        return {"type": "mixed_language", "penalty": 0.4}
    
    # 3. JSON 值中包含英文句子
    if parsed_json:
        def has_english_sentence(obj):
            if isinstance(obj, str):
                return bool(re.search(r'\b[a-zA-Z]{3,}(?:\s+[a-zA-Z]{3,}){2,}\b', obj))
            elif isinstance(obj, dict):
                return any(has_english_sentence(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(has_english_sentence(v) for v in obj)
            return False
        if has_english_sentence(parsed_json):
            return {"type": "json_value_pollution", "penalty": 0.35}
    
    return None


def compute_ngram_repetition(text: str, ngram_size: int = 4) -> float:
    """计算字符级 n-gram 重复率"""
    chars = list(text.replace(" ", "").replace("\n", ""))
    if len(chars) < ngram_size:
        return 0.0
    
    ngrams = set()
    total = 0
    for i in range(len(chars) - ngram_size + 1):
        ng = tuple(chars[i:i + ngram_size])
        ngrams.add(ng)
        total += 1
    
    return 1 - len(ngrams) / total if total > 0 else 0.0


def check_content_issues(text: str, ground_truth: str = "") -> Optional[Dict]:
    """检测内容问题（重复 + 长度 + 双重输出）"""
    # 1. 连续重复
    if re.search(r'(.{10,}?)\1{2,}', text):
        return {"type": "repetition_consecutive", "penalty": 0.9}  # 从 0.7 提高到 0.9
    
    # 2. n-gram 重复
    rep_ratio = compute_ngram_repetition(text, ngram_size=4)
    if rep_ratio > 0.25:  # 从 0.30 降低到 0.25，更早触发
        penalty = min((rep_ratio - 0.25) * 1.5, 0.8)  # 系数从 1.2 提高到 1.5，上限从 0.6 提高到 0.8
        return {"type": "repetition_ngram", "ratio": round(rep_ratio, 3), "penalty": round(penalty, 3)}
    
    # 3. 双重输出（JSON 前有大段文本）
    json_start = text.find('{')
    if json_start > 60:
        prefix = text[:json_start].strip()
        if len(prefix) > 50:
            return {"type": "double_output", "prefix_len": len(prefix), "penalty": 0.6}
    
    # 4. 时间戳泄露
    if re.search(r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]', text):
        return {"type": "timestamp_leak", "penalty": 0.3}
    
    # 5. 长度异常（增强版）
    if ground_truth and len(ground_truth) > 0:
        ratio = len(text) / len(ground_truth)
        # 过长惩罚：更早触发，更强惩罚
        if ratio > 1.3:  # 从 1.5 降低到 1.3
            penalty = min((ratio - 1.3) * 0.5, 1)  # 系数从 0.2 提高到 0.3，上限从 0.6 提高到 0.7
            return {"type": "too_long", "ratio": round(ratio, 2), "penalty": round(penalty, 3)}
        elif ratio < 0.3:
            return {"type": "too_short", "ratio": round(ratio, 2), "penalty": 0.3}
    
    return None


def compute_format_score(solution_str: str, ground_truth: str = "") -> Dict[str, Any]:
    """
    计算格式奖励分数（通用版 v4）
    
    核心设计：
    - 以 ground_truth 为参照
    - 多个问题可以累加惩罚
    - 增加惩罚力度，防止 reward hacking
    
    Returns:
        {"score": float, "penalties": dict}
    """
    if not solution_str or len(solution_str.strip()) < 3:
        return {"score": -1.0, "penalties": {"empty_output": True}}
    
    score = 0.0
    penalties = {}
    
    # 1. JSON 格式一致性检测（以 ground_truth 为参照）
    json_issue, parsed_json = check_json_consistency(solution_str, ground_truth)
    if json_issue:
        score -= json_issue["penalty"]
        penalties["format"] = json_issue
    elif parsed_json:
        score += 0.05  # JSON 正确给小奖励
    
    # 2. 语言污染检测（可与其他惩罚累加）
    lang_issue = check_language_pollution(solution_str, parsed_json)
    if lang_issue:
        score -= lang_issue["penalty"]
        penalties["language"] = lang_issue
    
    # 3. 内容问题检测（可与其他惩罚累加）
    content_issue = check_content_issues(solution_str, ground_truth)
    if content_issue:
        score -= content_issue["penalty"]
        penalties["content"] = content_issue
    
    # 3.5 长度接近奖励：鼓励输出长度接近 ground_truth
    if ground_truth and len(ground_truth) > 0 and "content" not in penalties:
        ratio = len(solution_str) / len(ground_truth)
        if 0.8 <= ratio <= 1.2:  # 长度在 80%-120% 之间
            length_bonus = 0.02  # 给予小奖励
            score += length_bonus
            penalties["length_bonus"] = {"ratio": round(ratio, 2), "bonus": length_bonus}
    
    # 4. 额外检测：即使 JSON 正确，也检测重复和语言问题
    # 这样可以防止模型学会"格式正确但内容有问题"的 hack
    if parsed_json and "content" not in penalties:
        # 检测 JSON 值中的重复
        json_str = json.dumps(parsed_json, ensure_ascii=False)
        rep_ratio = compute_ngram_repetition(json_str, ngram_size=4)
        if rep_ratio > 0.30:  # 从 0.35 降低到 0.30
            penalty = min((rep_ratio - 0.30) * 1.5, 0.8)  # 系数从 1.2 提高到 1.5，上限从 0.6 提高到 0.8
            score -= penalty
            penalties["json_repetition"] = {"ratio": round(rep_ratio, 3), "penalty": round(penalty, 3)}
    
    # 限制分数范围，但允许更大的负值惩罚
    return {"score": max(min(score, 0.1), -1.5), "penalties": penalties}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    """兼容 verl reward 系统的接口"""
    return compute_format_score(solution_str, ground_truth)["score"]
