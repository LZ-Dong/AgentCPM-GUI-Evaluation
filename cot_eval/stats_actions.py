#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

# 使用非交互式后端，避免服务器无显示环境时报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 用户请替换为实际模块路径（示例：from mypkg.extractors import extract_gt_action）
try:
    # 请将下一行替换为你的实际导入路径
    from annotation import extract_gt_action  # type: ignore
except Exception:
    extract_gt_action = None  # type: ignore


# --------- 工具函数 ---------

def sanitize_name(name: str) -> str:
    """将数据集名称转换为安全的文件名。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "dataset"


def warn(msg: str) -> None:
    sys.stderr.write(f"Warning: {msg}\n")


def parse_dataset_arg(s: str) -> Tuple[str, str]:
    """解析 NAME=PATH 形式的参数。"""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--dataset 需要 NAME=PATH 形式, 收到: {s}")
    name, path = s.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--dataset 中 NAME 为空: {s}")
    if not path:
        raise argparse.ArgumentTypeError(f"--dataset 中 PATH 为空: {s}")
    return name, path


def format_ratio(n: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(n / total) * 100:.2f}%"


def maybe_exit_if_no_import():
    """若用户未替换 import，则给出友好提示并退出。"""
    if extract_gt_action is None:
        sys.stderr.write(
            "错误：未找到 extract_gt_action。请在文件顶部替换为你的实际导入路径：\n"
            "  from <你的模块路径> import extract_gt_action\n"
        )
        sys.exit(2)


# --------- 核心统计逻辑 ---------

def process_jsonl_file(path: str) -> Tuple[Counter, int, int]:
    """
    读取 JSONL 文件并统计 action_type。
    返回: (counts, total, invalid)
      - counts: 各 action_type 计数（包含 __INVALID__ 和 __UNKNOWN__）
      - total: 总样本数（按行计数）
      - invalid: 无效样本数（解析失败或调用失败）
    """
    counts: Counter = Counter()
    total = 0
    invalid = 0

    # 逐行读取，行内异常独立处理
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                total += 1
                line = raw_line.strip()
                if not line:
                    invalid += 1
                    counts["__INVALID__"] += 1
                    continue
                try:
                    example = json.loads(line)
                except Exception:
                    invalid += 1
                    counts["__INVALID__"] += 1
                    continue

                # 调用用户提供的解析函数，仅使用 action_type
                try:
                    _, action_type = extract_gt_action(example)  # type: ignore
                except Exception:
                    invalid += 1
                    counts["__INVALID__"] += 1
                    continue

                # 未识别/无效的 action_type 归为 __UNKNOWN__
                if not isinstance(action_type, str) or not action_type.strip():
                    action_type = "__UNKNOWN__"

                counts[action_type] += 1
    except FileNotFoundError:
        raise
    except Exception as e:
        # 文件级别错误也尽量提示
        warn(f"读取文件时发生异常: {path} -> {e}")

    return counts, total, invalid


def write_dataset_outputs(
    outdir: str,
    dataset_name: str,
    counts: Counter,
    total: int,
) -> Tuple[str, str, str]:
    """
    输出每个数据集的 CSV / JSON / PNG。
    返回三个文件路径 (csv_path, json_path, png_path)。
    """
    os.makedirs(outdir, exist_ok=True)
    safe_name = sanitize_name(dataset_name)

    # 按计数降序排序，名称升序作为次序
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    csv_path = os.path.join(outdir, f"{safe_name}_stats.csv")
    json_path = os.path.join(outdir, f"{safe_name}_stats.json")
    png_path = os.path.join(outdir, f"{safe_name}_bar.png")

    # 写 CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["action_type", "count", "ratio"])
        for action_type, cnt in items:
            writer.writerow([action_type, cnt, format_ratio(cnt, total)])

    # 写 JSON（列表形式，列与 CSV 一致）
    json_records = [
        {"action_type": action_type, "count": cnt, "ratio": format_ratio(cnt, total)}
        for action_type, cnt in items
    ]
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(json_records, fp, ensure_ascii=False, indent=2)

    # 画条形图（matplotlib，不使用 seaborn）
    if items:
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4.5))
        bars = ax.bar(range(len(labels)), values, color="#4C78A8")
        ax.set_title(f"Action-Type Counts: {dataset_name}")
        ax.set_xlabel("action_type")
        ax.set_ylabel("count")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        # 在柱顶标注计数
        for rect, val in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(val),
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    else:
        # 空数据集也生成一个空图，避免后续流程报错
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Action-Type Counts: {dataset_name} (empty)")
        ax.set_xlabel("action_type")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

    return csv_path, json_path, png_path


def write_overview_csv(
    outdir: str,
    datasets: Dict[str, Dict[str, Any]],
) -> str:
    """
    写总览 CSV：action_type 行，按数据集并排列出 count 与 ratio。
    datasets: name -> {"counts": Counter, "total": int}
    """
    os.makedirs(outdir, exist_ok=True)
    overview_path = os.path.join(outdir, "overview_stats.csv")

    # 汇总所有 action_type
    all_action_types = set()
    for info in datasets.values():
        all_action_types.update(info["counts"].keys())

    # 按所有数据集的总和降序排序
    action_type_totals: Dict[str, int] = {}
    for at in all_action_types:
        action_type_totals[at] = sum(info["counts"].get(at, 0) for info in datasets.values())
    ordered_action_types = sorted(all_action_types, key=lambda k: (-action_type_totals[k], k))

    # 写 CSV
    header = ["action_type"]
    for name in datasets.keys():
        header.append(f"{name}:count")
        header.append(f"{name}:ratio")

    with open(overview_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for at in ordered_action_types:
            row = [at]
            for name, info in datasets.items():
                cnt = int(info["counts"].get(at, 0))
                total = int(info["total"])
                row.append(cnt)
                row.append(format_ratio(cnt, total))
            writer.writerow(row)

    return overview_path


def print_overview_table(datasets: Dict[str, Dict[str, Any]]) -> None:
    """
    终端打印对齐的汇总表（等宽字体友好），以及每个数据集的总样本数与无效样本数。
    """
    # 汇总所有 action_type
    all_action_types = set()
    for info in datasets.values():
        all_action_types.update(info["counts"].keys())

    # 排序：按总计数降序
    action_type_totals: Dict[str, int] = {}
    for at in all_action_types:
        action_type_totals[at] = sum(info["counts"].get(at, 0) for info in datasets.values())
    ordered_action_types = sorted(all_action_types, key=lambda k: (-action_type_totals[k], k))

    # 构建表格数据
    headers = ["action_type"]
    for name in datasets.keys():
        headers.extend([f"{name}:count", f"{name}:ratio"])

    table: List[List[str]] = []
    for at in ordered_action_types:
        row: List[str] = [at]
        for name, info in datasets.items():
            cnt = int(info["counts"].get(at, 0))
            total = int(info["total"])
            row.append(str(cnt))
            row.append(format_ratio(cnt, total))
        table.append(row)

    # 计算列宽
    col_widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # 打印表头
    line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(line)
    print(sep)

    # 打印行
    for row in table:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))

    # 打印 Totals/Invalid 概览
    print()
    print("Dataset totals:")
    for name, info in datasets.items():
        print(f"- {name}: total={info['total']}, invalid={info['invalid']}")


# --------- 主函数/CLI ---------

def main(argv: List[str]) -> int:
    maybe_exit_if_no_import()

    parser = argparse.ArgumentParser(
        description="统计 JSONL 测试集的 Action-Type 分布，并导出 CSV/JSON/PNG 与总览。"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_arg,
        required=True,
        help="数据集配置，格式：NAME=/path/to/all.jsonl，可重复多次",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="输出目录",
    )
    args = parser.parse_args(argv)

    # 去重检查：不同数据集使用了相同路径则告警，但仍各自统计
    path_to_names: Dict[str, List[str]] = defaultdict(list)
    for name, path in args.dataset:
        path_to_names[os.path.abspath(path)].append(name)
    for p, names in path_to_names.items():
        if len(names) > 1:
            warn(f"以下数据集使用了相同的文件路径 {p} ：{', '.join(names)}")

    # 逐数据集处理
    datasets_result: Dict[str, Dict[str, Any]] = {}
    os.makedirs(args.outdir, exist_ok=True)

    for name, path in args.dataset:
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            warn(f"文件不存在，跳过数据集 {name}: {abs_path}")
            continue

        counts, total, invalid = process_jsonl_file(abs_path)
        datasets_result[name] = {"counts": counts, "total": total, "invalid": invalid}

        # 输出每个数据集的明细
        write_dataset_outputs(args.outdir, name, counts, total)

    if not datasets_result:
        sys.stderr.write("没有可用的数据集统计结果，程序退出。\n")
        return 1

    # 写总览 CSV 并打印终端表格
    write_overview_csv(args.outdir, datasets_result)
    print_overview_table(datasets_result)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
"""
python stats_actions.py \
  --dataset AITZ=/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/aitz_test/all.jsonl \
  --dataset ChineseApp=/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/chinese_app_test/all.jsonl \
  --dataset AndroidControl=/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/android_control_high_test/all.jsonl \
  --outdir ./stats_out
"""