## GUI-Owl-7B

### Inference

```bash
# aitz_test
python run_predict_gui_owl_vllm.py --model_path ../model/GUI-Owl-7B --output_dir ./eval_results/GUI-Owl-7B/aitz_test --data_name aitz_test

# gui_odyssey_test
python run_predict_gui_owl_vllm.py --model_path ../model/GUI-Owl-7B --output_dir ./eval_results/GUI-Owl-7B/gui_odyssey_test --data_name gui_odyssey_test

# chinese_app_test
python run_predict_gui_owl_vllm.py --model_path ../model/GUI-Owl-7B --output_dir ./eval_results/GUI-Owl-7B/chinese_app_test --data_name chinese_app_test

# android_control_high_test
python run_predict_gui_owl_vllm.py --model_path ../model/GUI-Owl-7B --output_dir ./eval_results/GUI-Owl-7B/android_control_high_test --data_name android_control_high_test

# android_control_low_test
python run_predict_gui_owl.py --model_path ../model/GUI-Owl-7B --output_dir ./eval_results/GUI-Owl-7B/android_control_low_test --data_name android_control_low_test
```

### Eval

```bash
# aitz_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-7B/aitz_test/all.jsonl --output_dir ./eval_results/GUI-Owl-7B/aitz_test/results --data_name aitz_test

# gui_odyssey_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-7B/gui_odyssey_test/all.jsonl --output_dir ./eval_results/GUI-Owl-7B/gui_odyssey_test/results --data_name gui_odyssey_test

# chinese_app_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-7B/chinese_app_test/all.jsonl --output_dir ./eval_results/GUI-Owl-7B/chinese_app_test/results --data_name chinese_app_test

# android_control_high_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-7B/android_control_high_test/all.jsonl --output_dir ./eval_results/GUI-Owl-7B/android_control_high_test/results --data_name android_control_high_test --eval_android_control

# android_control_low_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-7B/android_control_low_test/all.jsonl --output_dir ./eval_results/GUI-Owl-7B/android_control_low_test/results --data_name android_control_low_test --eval_android_control
```

## GUI-Owl-32B

### Inference

```bash
# aitz_test
python run_predict_gui_owl_vllm.py --model_path ../model/GUI-Owl-32B --output_dir ./eval_results/GUI-Owl-32B/aitz_test --data_name aitz_test

# chinese_app_test
python run_predict_gui_owl_vllm.py --model_path ../model/GUI-Owl-32B --output_dir ./eval_results/GUI-Owl-32B/chinese_app_test --data_name chinese_app_test

# android_control_high_test
python run_predict_gui_owl_vllm.py --model_path ../model/GUI-Owl-32B --output_dir ./eval_results/GUI-Owl-32B/android_control_high_test --data_name android_control_high_test
```

### Eval

```bash
# aitz_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-32B/aitz_test/all.jsonl --output_dir ./eval_results/GUI-Owl-32B/aitz_test/results --data_name aitz_test

# chinese_app_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-32B/chinese_app_test/all.jsonl --output_dir ./eval_results/GUI-Owl-32B/chinese_app_test/results --data_name chinese_app_test

# android_control_high_test
python run_eval_agent.py --input_path ./eval_results/GUI-Owl-32B/android_control_high_test/all.jsonl --output_dir ./eval_results/GUI-Owl-32B/android_control_high_test/results --data_name android_control_high_test --eval_android_control
```