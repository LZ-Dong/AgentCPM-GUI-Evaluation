#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
更新 annotations.jsonl 的 em 与四象限指标 q1~q4。
规则：若 gta 为 "NA"（大小写均可）或缺失，则该样本不计入 q1~q4（全部置 0，且不纳入象限计数）。

四象限（供参考）：
Q1: EM_t=1, GTA_t=1  (Ideal)
Q2: EM_t=0, GTA_t=1  (Execution Gap, EG)
Q3: EM_t=0, GTA_t=0  (Both Wrong)
Q4: EM_t=1, GTA_t=0  (Reasoning Gap, RG)
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple

def load_em_map(result_json_path: str) -> Dict[str, int]:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    em_map: Dict[str, int] = {}
    for item in data:
        eid = str(item.get("episode_id", ""))
        sid = item.get("step_id", None)
        if eid == "" or sid is None:
            continue
        key = f"{eid}:{sid}"
        em_map[key] = 1 if bool(item.get("exact_match", False)) else 0
    return em_map

def parse_gta(v: Any) -> Tuple[int, bool]:
    """
    解析 gta：
      返回 (gta_val, is_valid)
      gta_val ∈ {0,1}；当 gta 为 'NA'/None/'' 等无效值时，is_valid=False，gta_val 置 0。
    """
    if v is None:
        return 0, False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("na", ""):
            return 0, False
        if s in ("1", "true"):
            return 1, True
        if s in ("0", "false"):
            return 0, True
        # 其他字符串意外值：按无效处理以避免误计入
        return 0, False
    if v in (1, True):
        return 1, True
    if v in (0, False):
        return 0, True
    return 0, False

def quadrants(em: int, gta: int) -> Dict[str, int]:
    return {
        "q1": 1 if (em == 1 and gta == 1) else 0,
        "q2": 1 if (em == 0 and gta == 1) else 0,
        "q3": 1 if (em == 0 and gta == 0) else 0,
        "q4": 1 if (em == 1 and gta == 0) else 0,
    }

def update_annotations(ann_path: str, em_map: Dict[str, int], gta_map: Dict[str, int]) -> None:
    tmp_path = ann_path + ".tmp"

    n_total = 0
    n_found_em = 0
    n_gta_na = 0
    cnt = {"q1":0, "q2":0, "q3":0, "q4":0}

    with open(ann_path, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            obj = json.loads(line)

            key = str(obj.get("id", ""))
            em_val = em_map.get(key, 0)
            gta_sys = gta_map.get(key, 0)
            if key in em_map:
                n_found_em += 1

            gta_raw = obj.get("gta", None)
            gta_val, gta_valid = parse_gta(gta_raw)

            # 计算 q1~q4（若 gta 无效，则全部置 0 且不计入统计）
            if gta_valid:
                qs = quadrants(em_val, gta_val)
                for qk in ("q1","q2","q3","q4"):
                    cnt[qk] += qs[qk]
            else:
                n_gta_na += 1
                qs = {"q1":0, "q2":0, "q3":0, "q4":0}

            # 写入新字段（保留其他字段）
            obj["em"] = em_val
            obj["gta_sys"] = gta_sys  # 来自模型的 gta 预测
            obj.update(qs)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(tmp_path, ann_path)

    print(f"[Done] 总样本: {n_total} | 匹配到 EM: {n_found_em} | gta=NA/无效: {n_gta_na}")
    print(f"（象限统计已排除 gta=NA） q1={cnt['q1']}, q2={cnt['q2']}, q3={cnt['q3']}, q4={cnt['q4']}")

def main():
    parser = argparse.ArgumentParser(description="Update annotations.jsonl with EM and quadrant indicators (excluding gta=NA).")
    parser.add_argument(
        "--result_json",
        default="/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/aitz_test/results/result.json",
        help="eval 产出的 result.json 路径"
    )
    parser.add_argument(
        "--cot_json",
        default="/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI_cot/aitz_test/results/result.json",
        help="COT 产出的 result.json 路径"
    )
    parser.add_argument(
        "--annotations_jsonl",
        default="/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/annotations.jsonl",
        help="人工标注的 annotations.jsonl 路径"
    )
    args = parser.parse_args()

    em_map = load_em_map(args.result_json)
    gta_map = load_em_map(args.cot_json)  # 复用函数，gta_map 结构同 em_map
    update_annotations(args.annotations_jsonl, em_map, gta_map)

if __name__ == "__main__":
    main()
