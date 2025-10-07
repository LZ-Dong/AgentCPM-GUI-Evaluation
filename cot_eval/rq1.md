# AgentCPM-GUI + AITZ
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/AgentCPM-GUI/aitz_test/annotations_zzq.jsonl data/AgentCPM-GUI/aitz_test/annotations_ysb.jsonl --policy strict --out-jsonl data/AgentCPM-GUI/aitz_test/merged_strict.jsonl --out-csv data/AgentCPM-GUI/aitz_test/merged_strict.csv --conflicts-jsonl data/AgentCPM-GUI/aitz_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/AgentCPM-GUI/aitz_test/merged_strict.jsonl --conflicts data/AgentCPM-GUI/aitz_test/conflicts_strict.jsonl --out-jsonl data/AgentCPM-GUI/aitz_test/gta_strict_clean.jsonl --out-csv data/AgentCPM-GUI/aitz_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/aitz_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI_cot/aitz_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/gta_strict_clean.jsonl
```

# AgentCPM-GUI + chinese_app
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/AgentCPM-GUI/chinese_app_test/annotations_zzq.jsonl data/AgentCPM-GUI/chinese_app_test/annotations_dlz.jsonl --policy strict --out-jsonl data/AgentCPM-GUI/chinese_app_test/merged_strict.jsonl --out-csv data/AgentCPM-GUI/chinese_app_test/merged_strict.csv --conflicts-jsonl data/AgentCPM-GUI/chinese_app_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/AgentCPM-GUI/chinese_app_test/merged_strict.jsonl --conflicts data/AgentCPM-GUI/chinese_app_test/conflicts_strict.jsonl --out-jsonl data/AgentCPM-GUI/chinese_app_test/gta_strict_clean.jsonl --out-csv data/AgentCPM-GUI/chinese_app_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/chinese_app_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI_cot/chinese_app_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/chinese_app_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/chinese_app_test/gta_strict_clean.jsonl
```

# AgentCPM-GUI + android_control_high_test
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/AgentCPM-GUI/android_control_high_test/annotations_shy.jsonl data/AgentCPM-GUI/android_control_high_test/annotations_dlz.jsonl --policy strict --out-jsonl data/AgentCPM-GUI/android_control_high_test/merged_strict.jsonl --out-csv data/AgentCPM-GUI/android_control_high_test/merged_strict.csv --conflicts-jsonl data/AgentCPM-GUI/android_control_high_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/AgentCPM-GUI/android_control_high_test/merged_strict.jsonl --conflicts data/AgentCPM-GUI/android_control_high_test/conflicts_strict.jsonl --out-jsonl data/AgentCPM-GUI/android_control_high_test/gta_strict_clean.jsonl --out-csv data/AgentCPM-GUI/android_control_high_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/android_control_high_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI_cot/android_control_high_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/android_control_high_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/android_control_high_test/gta_strict_clean.jsonl
```

# UI-TARS-1.5-7B + AITZ
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/UI-TARS-1.5-7B/aitz_test/annotations_zzq.jsonl data/UI-TARS-1.5-7B/aitz_test/annotations_ysb.jsonl --policy strict --out-jsonl data/UI-TARS-1.5-7B/aitz_test/merged_strict.jsonl --out-csv data/UI-TARS-1.5-7B/aitz_test/merged_strict.csv --conflicts-jsonl data/UI-TARS-1.5-7B/aitz_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/UI-TARS-1.5-7B/aitz_test/merged_strict.jsonl --conflicts data/UI-TARS-1.5-7B/aitz_test/conflicts_strict.jsonl --out-jsonl data/UI-TARS-1.5-7B/aitz_test/gta_strict_clean.jsonl --out-csv data/UI-TARS-1.5-7B/aitz_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B/aitz_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B_cot/aitz_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/aitz_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/aitz_test/gta_strict_clean.jsonl
```

# UI-TARS-1.5-7B + chinese_app
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/UI-TARS-1.5-7B/chinese_app_test/annotations_zzq.jsonl data/UI-TARS-1.5-7B/chinese_app_test/annotations_dlz.jsonl --policy strict --out-jsonl data/UI-TARS-1.5-7B/chinese_app_test/merged_strict.jsonl --out-csv data/UI-TARS-1.5-7B/chinese_app_test/merged_strict.csv --conflicts-jsonl data/UI-TARS-1.5-7B/chinese_app_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/UI-TARS-1.5-7B/chinese_app_test/merged_strict.jsonl --conflicts data/UI-TARS-1.5-7B/chinese_app_test/conflicts_strict.jsonl --out-jsonl data/UI-TARS-1.5-7B/chinese_app_test/gta_strict_clean.jsonl --out-csv data/UI-TARS-1.5-7B/chinese_app_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B/chinese_app_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B_cot/chinese_app_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/chinese_app_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/chinese_app_test/gta_strict_clean.jsonl
```

# UI-TARS-1.5-7B + android_control_high_test
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/UI-TARS-1.5-7B/android_control_high_test/annotations_shy.jsonl data/UI-TARS-1.5-7B/android_control_high_test/annotations_dlz.jsonl --policy strict --out-jsonl data/UI-TARS-1.5-7B/android_control_high_test/merged_strict.jsonl --out-csv data/UI-TARS-1.5-7B/android_control_high_test/merged_strict.csv --conflicts-jsonl data/UI-TARS-1.5-7B/android_control_high_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/UI-TARS-1.5-7B/android_control_high_test/merged_strict.jsonl --conflicts data/UI-TARS-1.5-7B/android_control_high_test/conflicts_strict.jsonl --out-jsonl data/UI-TARS-1.5-7B/android_control_high_test/gta_strict_clean.jsonl --out-csv data/UI-TARS-1.5-7B/android_control_high_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B/android_control_high_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B_cot/android_control_high_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/android_control_high_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/android_control_high_test/gta_strict_clean.jsonl
```

# GUI-Owl-7B + AITZ
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/GUI-Owl-7B/aitz_test/annotations_zzq.jsonl data/GUI-Owl-7B/aitz_test/annotations_ysb.jsonl --policy strict --out-jsonl data/GUI-Owl-7B/aitz_test/merged_strict.jsonl --out-csv data/GUI-Owl-7B/aitz_test/merged_strict.csv --conflicts-jsonl data/GUI-Owl-7B/aitz_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/GUI-Owl-7B/aitz_test/merged_strict.jsonl --conflicts data/GUI-Owl-7B/aitz_test/conflicts_strict.jsonl --out-jsonl data/GUI-Owl-7B/aitz_test/gta_strict_clean.jsonl --out-csv data/GUI-Owl-7B/aitz_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B/aitz_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B_cot/aitz_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/aitz_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/aitz_test/gta_strict_clean.jsonl
```

# GUI-Owl-7B + chinese_app
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/GUI-Owl-7B/chinese_app_test/annotations_zzq.jsonl data/GUI-Owl-7B/chinese_app_test/annotations_dlz.jsonl --policy strict --out-jsonl data/GUI-Owl-7B/chinese_app_test/merged_strict.jsonl --out-csv data/GUI-Owl-7B/chinese_app_test/merged_strict.csv --conflicts-jsonl data/GUI-Owl-7B/chinese_app_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/GUI-Owl-7B/chinese_app_test/merged_strict.jsonl --conflicts data/GUI-Owl-7B/chinese_app_test/conflicts_strict.jsonl --out-jsonl data/GUI-Owl-7B/chinese_app_test/gta_strict_clean.jsonl --out-csv data/GUI-Owl-7B/chinese_app_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B/chinese_app_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B_cot/chinese_app_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/chinese_app_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/chinese_app_test/gta_strict_clean.jsonl
```

# GUI-Owl-7B + android_control_high_test
```bash
# 两名独立评审标注汇总
python merge_jsonl.py data/GUI-Owl-7B/android_control_high_test/annotations_shy.jsonl data/GUI-Owl-7B/android_control_high_test/annotations_dlz.jsonl --policy strict --out-jsonl data/GUI-Owl-7B/android_control_high_test/merged_strict.jsonl --out-csv data/GUI-Owl-7B/android_control_high_test/merged_strict.csv --conflicts-jsonl data/GUI-Owl-7B/android_control_high_test/conflicts_strict.jsonl
# 冲突条目由第三名评审决定
python resolve_conflicts.py --merged data/GUI-Owl-7B/android_control_high_test/merged_strict.jsonl --conflicts data/GUI-Owl-7B/android_control_high_test/conflicts_strict.jsonl --out-jsonl data/GUI-Owl-7B/android_control_high_test/gta_strict_clean.jsonl --out-csv data/GUI-Owl-7B/android_control_high_test/gta_strict_clean.csv
# 更新em和gta_sys
python update.py \
--result_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B/android_control_high_test/results/result.json \
--cot_json /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B_cot/android_control_high_test/results/result.json \
--annotations_jsonl /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/android_control_high_test/gta_strict_clean.jsonl
# 评估gta_sys和人类标注gta的一致性
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/android_control_high_test/gta_strict_clean.jsonl
```