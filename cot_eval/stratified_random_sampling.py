import math
import json
import random
import os
from typing import Dict, List, Tuple
from annotation import extract_gt_action

extract_key = lambda x: f"{x['episode_id']}:{x['step_id']}"

# Configuration constants
DEFAULT_SEED = 2025
DEFAULT_SAMPLE_COUNT = 200
MIN_SAMPLES_PER_STRATUM = 5

# Dataset configurations
DATASET_CONFIGS = {
    "aitz": {
        "stats": {"CLICK": 2736, "STOP": 504, "SCROLL": 601, "INPUT": 500, "PRESS": 383, "LONG_POINT": 0},
        "datasets": [
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/aitz_test/all.jsonl",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B/aitz_test/all.jsonl",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B/aitz_test/all.jsonl",
        ],
        "output_dirs": [
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/aitz_test",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/aitz_test",
        ]
    },
    "cagui": {
        "stats": {"CLICK": 3237, "STOP": 600, "SCROLL": 79, "INPUT": 574, "PRESS": 0, "LONG_POINT": 25},
        "datasets": [
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/chinese_app_test/all.jsonl",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B/chinese_app_test/all.jsonl",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B/chinese_app_test/all.jsonl",
        ],
        "output_dirs": [
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/chinese_app_test",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/chinese_app_test",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/chinese_app_test",
        ]
    },
    "ac": {
        "stats": {"CLICK": 5504, "STOP": 1680, "SCROLL": 1297, "INPUT": 685, "PRESS": 372, "LONG_POINT": 0},
        "datasets": [
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AgentCPM-GUI/android_control_high_test/all.jsonl",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/UI-TARS-1.5-7B/android_control_high_test/all.jsonl",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/GUI-Owl-7B/android_control_high_test/all.jsonl",
        ],
        "output_dirs": [
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/android_control_high_test",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/android_control_high_test",
            "/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/android_control_high_test",
        ]
    }
}

def load_dataset(dataset_path: str) -> List[dict]:
    """Load dataset from jsonl file with error handling."""
    try:
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num} in {dataset_path}: {e}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset {dataset_path}: {e}")

def stratified_random_sampling(dataset_path: str, sampling_count: Dict[str, int], 
                             output_dir: str, seed: int = DEFAULT_SEED) -> List[str]:
    """
    Perform stratified random sampling on the dataset based on action type distribution.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_dataset(dataset_path)
    
    # Group data by action types
    action_groups = {}
    for item in data:
        try:
            _, action_type = extract_gt_action(item)
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(extract_key(item))
        except Exception as e:
            print(f"Warning: Skipping item due to action extraction error: {e}")
    
    # Shuffle and sample from each group
    key_list = []
    for action_type, keys in action_groups.items():
        random.shuffle(keys)
        sample_count = sampling_count.get(action_type, 0)
        if sample_count > len(keys):
            print(f"Warning: Requested {sample_count} samples for {action_type}, but only {len(keys)} available")
            sample_count = len(keys)
        key_list.extend(keys[:sample_count])
    
    return key_list

def save_sampled_data(dataset_path: str, key_list: List[str], output_path: str) -> None:
    """Save the sampled data to a new file with error handling."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    key_set = set(key_list)
    saved_count = 0
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    item = json.loads(line.strip())
                    key = extract_key(item)
                    if key in key_set:
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        saved_count += 1
                except json.JSONDecodeError:
                    continue
        
        print(f"Saved {saved_count} samples to {output_path}")
        
    except Exception as e:
        raise Exception(f"Error saving sampled data to {output_path}: {e}")

def calculate_sampling_count(stat: Dict[str, int], num_samples: int, 
                           k: int = MIN_SAMPLES_PER_STRATUM, seed: int = DEFAULT_SEED) -> Dict[str, int]:
    """Calculate the number of samples to draw from each stratum."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    
    random.seed(seed)

    # Only consider categories with count > 0
    present = {c: n for c, n in stat.items() if n > 0}
    
    if not present:
        raise ValueError("No categories with positive counts found")

    # Step 1: Guarantee minimum samples
    base = {c: min(k, n) for c, n in present.items()}
    M = sum(base.values())
    R = num_samples - M
    
    if R < 0:
        raise ValueError(f"Minimum samples {M} exceeds target {num_samples}. Reduce k or increase num_samples.")

    # Step 2: Calculate proportional allocation for remaining samples
    total_present = sum(present.values())
    q = {c: R * (present[c] / total_present) for c in present}

    # Step 3: Floor allocation
    a = {c: int(math.floor(q[c])) for c in present}
    leftover = R - sum(a.values())

    # Step 4: Largest remainder method
    remainders = [(c, q[c] - a[c]) for c in present]
    remainders.sort(key=lambda x: (-x[1], random.random()))

    t = {c: base[c] + a[c] for c in present}
    for i in range(leftover):
        c = remainders[i % len(remainders)][0]
        if t[c] < present[c]:
            t[c] += 1

    # Step 5: Ensure not exceeding original counts
    for c in t:
        t[c] = min(t[c], present[c])

    # Final adjustment if needed
    diff = num_samples - sum(t.values())
    if diff != 0:
        largest = max(t, key=lambda x: present[x])
        t[largest] = max(0, t[largest] + diff)

    actual_total = sum(t.values())
    if actual_total != num_samples:
        print(f"Warning: Actual samples {actual_total} != target {num_samples}")
    
    return t

def process_dataset_group(dataset_name: str, sample_count: int = DEFAULT_SAMPLE_COUNT, 
                         seed: int = DEFAULT_SEED) -> None:
    """Process a complete dataset group (all models for one dataset)."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    try:
        # Calculate sampling distribution
        sampling_count = calculate_sampling_count(config["stats"], sample_count, seed=seed)
        print(f"Sampling distribution for {dataset_name}: {sampling_count}")
        
        # Generate unified key list using the first dataset
        key_list = stratified_random_sampling(
            config["datasets"][0], 
            sampling_count, 
            config["output_dirs"][0], 
            seed
        )
        print(f"Generated {len(key_list)} keys for {dataset_name}")
        
        # Apply same key list to all datasets in the group
        for dataset_path, output_dir in zip(config["datasets"], config["output_dirs"]):
            output_path = os.path.join(output_dir, "sampled.jsonl")
            save_sampled_data(dataset_path, key_list, output_path)
            
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        raise

def main():
    """Main execution function."""
    try:
        # Process all dataset groups
        for dataset_name in DATASET_CONFIGS.keys():
            print(f"\nProcessing {dataset_name}...")
            process_dataset_group(dataset_name)
        
        print("\nAll datasets processed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    # main()
    print(calculate_sampling_count(DATASET_CONFIGS['aitz']['stats'], 200))
    print(calculate_sampling_count(DATASET_CONFIGS['cagui']['stats'], 200))
    print(calculate_sampling_count(DATASET_CONFIGS['ac']['stats'], 200))