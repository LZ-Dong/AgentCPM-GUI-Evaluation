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