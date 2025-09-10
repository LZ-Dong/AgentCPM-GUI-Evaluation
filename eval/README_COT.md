## AgentCPM-GUI_cot

### Inference

```bash
# aitz_test
python run_predict_cot.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/AgentCPM-GUI_cot/aitz_test --data_name aitz_test --cot_path eval_results/AgentCPM-GUI/aitz_test/all.jsonl
# chinese_app_test
python run_predict_cot.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/AgentCPM-GUI_cot/chinese_app_test --data_name chinese_app_test --cot_path eval_results/AgentCPM-GUI/chinese_app_test/all.jsonl
# android_control_high_test
python run_predict_cot.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/AgentCPM-GUI_cot/android_control_high_test --data_name android_control_high_test --cot_path eval_results/AgentCPM-GUI/android_control_high_test/all.jsonl
```

### Eval

```bash
# aitz_test
python run_eval_agent.py --input_path ./eval_results/AgentCPM-GUI_cot/aitz_test/all.jsonl --output_dir ./eval_results/AgentCPM-GUI_cot/aitz_test/results --data_name aitz_test
# chinese_app_test
python run_eval_agent.py --input_path ./eval_results/AgentCPM-GUI_cot/chinese_app_test/all.jsonl --output_dir ./eval_results/AgentCPM-GUI_cot/chinese_app_test/results --data_name chinese_app_test
# android_control_high_test
python run_eval_agent.py --input_path ./eval_results/AgentCPM-GUI_cot/android_control_high_test/all.jsonl --output_dir ./eval_results/AgentCPM-GUI_cot/android_control_high_test/results --data_name android_control_high_test
```

## UI-TARS-7B-SFT_cot
### Inference

```bash
# android_control_high_test
python run_predict_cot.py --model_path ../model/AgentCPM-GUI --output_dir ./eval_results/UI-TARS-7B-SFT_cot/android_control_high_test --data_name android_control_high_test --cot_path eval_results/UI-TARS-7B-SFT/android_control_high_test/all.jsonl
```
