import argparse, json, sys, csv
from typing import Dict, Any, Iterable, Optional

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[warn] skip invalid JSON: {path}:{ln}: {e}", file=sys.stderr)

def is_01(x: Any) -> bool:
    return x in (0, 1) or (isinstance(x, str) and x in ("0", "1"))

def to_int01(x: Any) -> Optional[int]:
    if x in (0, 1):
        return int(x)
    if isinstance(x, str) and x in ("0", "1"):
        return int(x)
    return None

def pick_final_from_record(r: Dict[str, Any]) -> Optional[int]:
    # Priorities inside merged file in case it already has a resolved/merged field.
    for key in ("resolved_gta", "merged_gta", "gta"):
        if key in r and is_01(r[key]):
            return to_int01(r[key])
    # Fall back: equal numeric annotator labels
    if "gta_a" in r and "gta_b" in r and is_01(r["gta_a"]) and is_01(r["gta_b"]):
        if to_int01(r["gta_a"]) == to_int01(r["gta_b"]):
            return to_int01(r["gta_a"])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="path to merged_strict.jsonl")
    ap.add_argument("--conflicts", required=True, help="path to conflicts_strict.jsonl")
    ap.add_argument("--out-jsonl", required=True, help="output clean jsonl (id,gta)")
    ap.add_argument("--out-csv", required=True, help="output clean csv (id,gta)")
    ap.add_argument("--fail-on-unresolved", action="store_true",
                    help="exit non-zero if any conflict remains NA")
    args = ap.parse_args()

    # Load manual resolutions
    resolved: Dict[str, int] = {}
    unresolved_ids = []
    total_conflicts = 0
    for r in read_jsonl(args.conflicts):
        if r.get("reason") == "conflict":
            total_conflicts += 1
        rid = str(r.get("id"))
        v = r.get("resolved_gta")
        vi = to_int01(v)
        if vi is not None:
            resolved[rid] = vi
        else:
            # Track unresolved conflicts only
            if r.get("reason") == "conflict":
                unresolved_ids.append(rid)

    if unresolved_ids:
        print(f"[info] unresolved conflicts: {len(unresolved_ids)}/{total_conflicts}", file=sys.stderr)

    # Build clean outputs
    clean_rows = []
    total = 0
    decided = 0
    used_manual = 0
    used_auto = 0

    for r in read_jsonl(args.merged):
        total += 1
        rid = str(r.get("id"))
        final = None
        if rid in resolved:
            final = resolved[rid]
            used_manual += 1
        else:
            final = pick_final_from_record(r)
            if final is not None:
                used_auto += 1
        if final is not None:
            decided += 1
            clean_rows.append({"id": rid, "gta": final})

    # Write outputs
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "gta"])
        for row in clean_rows:
            w.writerow([row["id"], row["gta"]])

    print(f"[done] total in merged: {total}", file=sys.stderr)
    print(f"[done] clean decided: {decided}", file=sys.stderr)
    print(f"[done] used manual: {used_manual}, used auto: {used_auto}", file=sys.stderr)
    if args.fail_on_unresolved and unresolved_ids:
        print("[error] unresolved conflicts remain. Use --fail-on-unresolved to enforce.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
python resolve_conflicts.py --merged data/AgentCPM-GUI/aitz_test/merged_strict.jsonl --conflicts data/AgentCPM-GUI/aitz_test/conflicts_strict.jsonl --out-jsonl data/AgentCPM-GUI/aitz_test/gta_strict_clean.jsonl --out-csv data/AgentCPM-GUI/aitz_test/gta_strict_clean.csv
"""