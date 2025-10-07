#!/usr/bin/env python3
import argparse
import json
import csv
from collections import defaultdict

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj["id"]
            data[_id] = {
                "gta": obj.get("gta"),
                "error_code": obj.get("error_code", "")
            }
    return data

def norm_gta(v):
    # Accept "0","1","NA", 0,1,None
    if v is None:
        return None
    if isinstance(v, int):
        return v if v in (0, 1) else None
    s = str(v).strip().upper()
    if s == "NA" or s == "":
        return None
    if s in ("0", "1"):
        return int(s)
    return None

def denorm_gta(v):
    return "NA" if v is None else int(v)

def resolve_labels(a, b, policy="strict"):
    """
    a,b: 0/1/None
    Returns: resolved, agree_exact, agree_numeric, needs_adj, reason
    """
    agree_exact = (a == b)
    agree_numeric = (a in (0,1) and b in (0,1) and a == b)

    if policy == "lenient":
        if a is None and b is None:
            return None, agree_exact, agree_numeric, False, "both_na"
        if a in (0,1) and b is None:
            return a, agree_exact, agree_numeric, False, "single_label"
        if a is None and b in (0,1):
            return b, agree_exact, agree_numeric, False, "single_label"
        if a in (0,1) and b in (0,1):
            if a == b:
                return a, True, True, False, "agree"
            else:
                return None, False, False, True, "conflict"
        # fallback
        return None, agree_exact, agree_numeric, True, "unknown"
    else:  # strict
        if a is None or b is None:
            # Any NA -> NA
            if a is None and b is None:
                return None, True, False, False, "both_na"
            else:
                return None, (a is None and b is None), False, False, "has_na"
        if a == b:
            return a, True, True, False, "agree"
        else:
            return None, False, False, True, "conflict"

def cohen_kappa(n00, n01, n10, n11):
    # Only among numeric pairs
    n = n00 + n01 + n10 + n11
    if n == 0:
        return None
    Po = (n00 + n11) / n
    p0 = ((n00 + n01) * (n00 + n10)) / (n * n)
    p1 = ((n10 + n11) * (n01 + n11)) / (n * n)
    Pe = p0 + p1
    if Pe == 1.0:
        return None
    return (Po - Pe) / (1 - Pe)

def main():
    ap = argparse.ArgumentParser(description="Merge two annotators' JSONL and compute agreement.")
    ap.add_argument("file_a", help="Path to annotations A (jsonl)")
    ap.add_argument("file_b", help="Path to annotations B (jsonl)")
    ap.add_argument("--out-jsonl", default="merged.jsonl", help="Output merged jsonl")
    ap.add_argument("--out-csv", default="merged.csv", help="Output merged csv")
    ap.add_argument("--conflicts-jsonl", default="conflicts.jsonl", help="Output conflicts jsonl")
    ap.add_argument("--policy", choices=["strict", "lenient"], default="strict", help="Merge policy")
    args = ap.parse_args()

    A = load_jsonl(args.file_a)
    B = load_jsonl(args.file_b)
    ids = sorted(set(A.keys()) | set(B.keys()))

    stats = defaultdict(int)
    # confusion matrix for numeric pairs: rows=A, cols=B
    n00 = n01 = n10 = n11 = 0

    merged_rows = []
    conflict_rows = []

    for _id in ids:
        ga_raw = A.get(_id, {}).get("gta")
        gb_raw = B.get(_id, {}).get("gta")
        ea = A.get(_id, {}).get("error_code", "MISSING" if _id not in A else "")
        eb = B.get(_id, {}).get("error_code", "MISSING" if _id not in B else "")

        ga = norm_gta(ga_raw)
        gb = norm_gta(gb_raw)

        resolved, agree_exact, agree_numeric, needs_adj, reason = resolve_labels(ga, gb, args.policy)

        # stats
        stats["total"] += 1
        if ga is None: stats["a_na"] += 1
        if gb is None: stats["b_na"] += 1
        if ga is None and gb is None: stats["both_na"] += 1
        if (ga in (0,1)) and (gb in (0,1)):
            stats["both_numeric"] += 1
            if ga == 0 and gb == 0: n00 += 1
            elif ga == 0 and gb == 1: n01 += 1
            elif ga == 1 and gb == 0: n10 += 1
            elif ga == 1 and gb == 1: n11 += 1
            if ga == gb:
                stats["agree_numeric"] += 1
        if agree_exact:
            stats["agree_exact"] += 1
        if needs_adj:
            stats["needs_adj"] += 1
        if reason == "has_na":
            stats["any_na"] += 1
        if reason == "conflict":
            stats["conflicts"] += 1

        row = {
            "id": _id,
            "gta_a": "NA" if ga is None else ga,
            "gta_b": "NA" if gb is None else gb,
            "error_code_a": ea,
            "error_code_b": eb,
            "resolved_gta": "NA" if resolved is None else resolved,
            "agree_exact": bool(agree_exact),
            "agree_numeric": bool(agree_numeric),
            "needs_adjudication": bool(needs_adj),
            "reason": reason,
            "policy": args.policy
        }
        merged_rows.append(row)
        if needs_adj or reason == "conflict":
            conflict_rows.append(row)

    # Write JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in merged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.conflicts_jsonl, "w", encoding="utf-8") as f:
        for r in conflict_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write CSV
    fieldnames = ["id","gta_a","gta_b","error_code_a","error_code_b","resolved_gta","agree_exact","agree_numeric","needs_adjudication","reason","policy"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    # Stats
    total = stats["total"]
    both_numeric = stats["both_numeric"]
    agree_numeric = stats["agree_numeric"]
    exact_agree = stats["agree_exact"]
    kappa = cohen_kappa(n00, n01, n10, n11)
    print("=== Merge Summary ===")
    print(f"Policy: {args.policy}")
    print(f"Total items: {total}")
    print(f"Both numeric: {both_numeric} ({(both_numeric/total*100):.2f}%)")
    print(f"Numeric agreement: {agree_numeric}/{both_numeric} ({(agree_numeric/max(both_numeric,1)*100):.2f}%)")
    if kappa is not None:
        print(f"Cohen's kappa (on numeric pairs): {kappa:.4f}")
    else:
        print("Cohen's kappa: N/A")
    print(f"Exact agreement (including NA): {exact_agree}/{total} ({(exact_agree/total*100):.2f}%)")
    print(f"Both NA: {stats['both_na']}")
    print(f"Any NA (strict meaning): {stats.get('any_na', 0)}")
    print(f"Conflicts (0 vs 1): {stats['conflicts']}")
    print(f"Needs adjudication: {stats['needs_adj']}")
    print(f"Output -> {args.out_jsonl}, {args.out_csv}, {args.conflicts_jsonl}")

if __name__ == "__main__":
    main()

# python merge_jsonl.py data/AgentCPM-GUI/aitz_test/annotations_zzq.jsonl data/AgentCPM-GUI/aitz_test/annotations_ysb.jsonl --policy strict --out-jsonl data/AgentCPM-GUI/aitz_test/merged_strict.jsonl --out-csv data/AgentCPM-GUI/aitz_test/merged_strict.csv --conflicts-jsonl data/AgentCPM-GUI/aitz_test/conflicts_strict.jsonl