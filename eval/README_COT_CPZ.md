# Evaluation of Chain-of-Thought (CoT) with CPZ on Android Control High Test Set
```bash
python run_predict_cot_cpz.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/UI-TARS-72B-DPO_cot/android_control_high_test --data_name android_control_high_test --cot_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AndroidControl_high_UI-TARS-72B-DPO_raw.json

python run_predict_cot_cpz.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/UI-TARS-72B-SFT_cot/android_control_high_test --data_name android_control_high_test --cot_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AndroidControl_high_UI-TARS-72B-SFT_raw.json

python run_predict_cot_cpz.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/UI-TARS-7B-DPO_cot/android_control_high_test --data_name android_control_high_test --cot_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AndroidControl_high_UI-TARS-7B-DPO_raw.json

python run_predict_cot_cpz.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/UI-TARS-7B-SFT_cot/android_control_high_test --data_name android_control_high_test --cot_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AndroidControl_high_UI-TARS-7B-SFT_raw.json

python run_predict_cot_cpz.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/UI-TARS-2B-SFT_cot/android_control_high_test --data_name android_control_high_test --cot_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AndroidControl_high_UI-TARS-2B-SFT_raw.json
```

```bash
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-DPO_cot/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-DPO_cot/android_control_high_test/results --data_name android_control_high_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-SFT_cot/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-SFT_cot/android_control_high_test/results --data_name android_control_high_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-72B-DPO_cot/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-72B-DPO_cot/android_control_high_test/results --data_name android_control_high_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-72B-SFT_cot/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-72B-SFT_cot/android_control_high_test/results --data_name android_control_high_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-2B-SFT_cot/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-2B-SFT_cot/android_control_high_test/results --data_name android_control_high_test
```