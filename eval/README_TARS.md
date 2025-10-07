## UI-TARS-7B-SFT
### Inference
```bash
# aitz_test
python run_predict_ui_tars.py --model_path ../model/UI-TARS-7B-SFT --output_dir ./eval_results/UI-TARS-7B-SFT/aitz_test --data_name aitz_test

# chinese_app_test
python run_predict_ui_tars.py --model_path ../model/UI-TARS-7B-SFT --output_dir ./eval_results/UI-TARS-7B-SFT/chinese_app_test --data_name chinese_app_test

# android_control_high_test
python run_predict_ui_tars.py --model_path ../model/UI-TARS-7B-SFT --output_dir ./eval_results/UI-TARS-7B-SFT/android_control_high_test --data_name android_control_high_test
```
### Eval
```bash
# aitz_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-SFT/aitz_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-SFT/aitz_test/results --data_name aitz_test

# chinese_app_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-SFT/chinese_app_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-SFT/chinese_app_test/results --data_name chinese_app_test

# android_control_high_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-SFT/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-SFT/android_control_high_test/results --data_name android_control_high_test --eval_android_control
```

## UI-TARS-7B-DPO
### Inference
```bash
# aitz_test
python run_predict_ui_tars.py --model_path ../model/UI-TARS-7B-DPO --output_dir ./eval_results/UI-TARS-7B-DPO/aitz_test --data_name aitz_test

# chinese_app_test
python run_predict_ui_tars.py --model_path ../model/UI-TARS-7B-DPO --output_dir ./eval_results/UI-TARS-7B-DPO/chinese_app_test --data_name chinese_app_test

# android_control_high_test
python run_predict_ui_tars.py --model_path ../model/UI-TARS-7B-DPO --output_dir ./eval_results/UI-TARS-7B-DPO/android_control_high_test --data_name android_control_high_test
```
### Eval
```bash
# aitz_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-DPO/aitz_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-DPO/aitz_test/results --data_name aitz_test

# chinese_app_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-DPO/chinese_app_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-DPO/chinese_app_test/results --data_name chinese_app_test

# android_control_high_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-7B-DPO/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-7B-DPO/android_control_high_test/results --data_name android_control_high_test --eval_android_control
```


## UI-TARS-1.5-7B
### Inference
```bash
# aitz_test
python run_predict_ui_tars1_5.py --model_path ../model/UI-TARS-1.5-7B --output_dir ./eval_results/UI-TARS-1.5-7B/aitz_test --data_name aitz_test

# chinese_app_test
python run_predict_ui_tars1_5.py --model_path ../model/UI-TARS-1.5-7B --output_dir ./eval_results/UI-TARS-1.5-7B/chinese_app_test --data_name chinese_app_test

# android_control_high_test
python run_predict_ui_tars1_5.py --model_path ../model/UI-TARS-1.5-7B --output_dir ./eval_results/UI-TARS-1.5-7B/android_control_high_test --data_name android_control_high_test
```
### Eval
```bash
# aitz_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-1.5-7B/aitz_test/all.jsonl --output_dir ./eval_results/UI-TARS-1.5-7B/aitz_test/results --data_name aitz_test

# chinese_app_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-1.5-7B/chinese_app_test/all.jsonl --output_dir ./eval_results/UI-TARS-1.5-7B/chinese_app_test/results --data_name chinese_app_test

# android_control_high_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-1.5-7B/android_control_high_test/all.jsonl --output_dir ./eval_results/UI-TARS-1.5-7B/android_control_high_test/results --data_name android_control_high_test
```

## UI-TARS-72B-SFT
### Inference
```bash
# aitz_test
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_predict_ui_tars_72b.py --model_path ../model/UI-TARS-72B-SFT --output_dir ./eval_results/UI-TARS-72B-SFT/aitz_test --data_name aitz_test --tensor_parallel_size 4 --max_gpu_mem 76GiB
```
### Eval
```bash
# aitz_test
python run_eval_agent.py --input_path ./eval_results/UI-TARS-72B-SFT/aitz_test/all.jsonl --output_dir ./eval_results/UI-TARS-72B-SFT/aitz_test/results --data_name aitz_test
```