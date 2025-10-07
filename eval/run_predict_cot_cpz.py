import sys
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import random
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir
import argparse
import logging
import time

DEVICES = [
    # "cuda:0", "cuda:1", "cuda:2", "cuda:3",
    # "cuda:4","cuda:5", "cuda:6", "cuda:7",
    # "cuda:0", "cuda:0", "cuda:0",
    "cuda:1", "cuda:1", "cuda:1",
    "cuda:2", "cuda:2", "cuda:2",
    "cuda:3", "cuda:3", "cuda:3",
    "cuda:4", "cuda:4", "cuda:4",
    ]

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if current_dir not in sys.path:
    sys.path.append(current_dir)

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)

ACTION_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema.json'), encoding="utf-8"))
items = list(ACTION_SCHEMA.items())
insert_index = 3
items.insert(insert_index, ("required", ["thought"])) # enable/disable thought by setting it to "required"/"optional"
ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = f'''# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''

EXTRACT_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))


_llm = None
_tokenizer = None

def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.bfloat16)
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def move_to(device):
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"


def run_episode(episode, msg,):
    global _llm,_tokenizer
    outputs = _llm.chat(image=None, msgs=msg, system_prompt=SYSTEM_PROMPT, tokenizer=_tokenizer, temperature=0.1,top_p=0.3,n=1,)
    episode["pred"] = extract_and_validate_json(outputs)
    return episode


def extract_and_validate_json(input_string):
    try:
        json_obj = json.loads(input_string)
        jsonschema.validate(json_obj, EXTRACT_SCHEMA)
        return json_obj
    except json.JSONDecodeError as e:
        print("Error, JSON is NOT valid.")
        return input_string
    except Exception as e:
        print(f"Error, JSON is NOT valid according to the schema.{input_string}", e)
        return input_string

def build_cot_dict(cot_path):
    """
    Build a dictionary from CoT sources mapping (episode_id, step_id) -> thought/query.

    Supported formats:
    1) JSON eval result with "detailed_results": list[...], each item containing:
       - episode_id, step_id, predicted_thought (preferred) or real_thought
       Example path: eval/eval_results/AndroidControl_high_UI-TARS-72B-DPO_raw.json
    2) JSONL (one JSON per line) from first-round inference where an entry contains:
       - episode_id, step_id, and either:
         a) pred.thought
         b) thought (top-level) or predicted_thought or real_thought

    Returns:
      cot_dict, total_entries, missing_cot_count
    """
    cot_dict = {}
    total_entries = 0
    missing_cot_count = 0

    if not cot_path or not os.path.exists(cot_path):
        print(f"Warning: CoT file not found at {cot_path}")
        return cot_dict, total_entries, missing_cot_count

    print(f"Building CoT dictionary from: {cot_path}")

    # Try to detect "eval JSON" format first (has 'detailed_results')
    try:
        with open(cot_path, 'r', encoding='utf-8') as f:
            head = f.read(8192)
            f.seek(0)
            if head.lstrip().startswith('{') and '"detailed_results"' in head:
                data = json.load(f)
                results = data.get('detailed_results', [])
                total_entries = len(results)
                for item in results:
                    episode_id = item.get('episode_id')
                    step_id = item.get('step_id') - 1
                    thought = (item.get('predicted_thought') or "").strip() if isinstance(item, dict) else ""
                    if episode_id is not None and step_id is not None and thought != "":
                        cot_dict[(episode_id, step_id)] = thought
                    else:
                        missing_cot_count += 1
                missing_ratio = (missing_cot_count / total_entries) if total_entries > 0 else 0
                print(f"Parsed eval results JSON: {len(cot_dict)} entries, {missing_cot_count} missing ({missing_ratio:.2%})")
                return cot_dict, total_entries, missing_cot_count
    except Exception as e:
        print(f"Info: Failed to parse as eval JSON with 'detailed_results', will try JSONL. Error: {e}")

    # Fallback: parse as JSONL
    try:
        with open(cot_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    total_entries += 1
                    episode_id = entry.get('episode_id')
                    step_id = entry.get('step_id')

                    # Prefer pred.thought if present
                    thought = None
                    pred_field = entry.get('pred')
                    if isinstance(pred_field, dict):
                        thought = pred_field.get('thought')

                    # Fallbacks: top-level fields commonly seen
                    if not thought:
                        thought = entry.get('thought') or entry.get('predicted_thought') or entry.get('real_thought')

                    thought = (thought or "").strip()

                    if episode_id is not None and step_id is not None and thought:
                        cot_dict[(episode_id, step_id)] = thought
                    else:
                        missing_cot_count += 1
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSONL line: {e}")
                    missing_cot_count += 1
                except Exception as e:
                    print(f"Error processing JSONL entry: {e}")
                    missing_cot_count += 1
    except Exception as e:
        print(f"Error opening CoT file as JSONL: {e}")
        return cot_dict, total_entries, missing_cot_count

    missing_ratio = (missing_cot_count / total_entries) if total_entries > 0 else 0
    print(f"CoT dictionary built: {len(cot_dict)} valid entries, {missing_cot_count} missing ({missing_ratio:.2%})")
    return cot_dict, total_entries, missing_cot_count

def load_image(episode, image_path, data_name, cot_dict=None):
    # resize the image proportionally so that the longer side is at most 1120
    def __resize__(origin_img):
        resolution = origin_img.size
        w,h = resolution
        max_line_res = 1120
        if max_line_res is not None:
            max_line = max_line_res
            if h > max_line:
                w = int(w * max_line / h)
                h = max_line
            if w > max_line:
                h = int(h * max_line / w)
                w = max_line
        img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
        return img

    image = Image.open(image_path).convert("RGB")
    image = __resize__(image)

    # Check if we have CoT available for this episode
    if cot_dict is not None:
        episode_id = episode.get('episode_id')
        step_id = episode.get('step_id')
        cot_key = (episode_id, step_id)
        
        if cot_key in cot_dict:
            # Use CoT as query
            query = cot_dict[cot_key]
        else:
            # Skip this episode if no CoT available
            return None
    else:
        # Original query selection logic
        if data_name == 'android_control_low_test':
            query = episode['low_instruction']
        else:
            query = episode['instruction']

    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                f"<Question>{query}</Question>\n当前屏幕截图：",
                image
            ]
        }
    )
    return (episode,messages)


def predict(args):
    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    print(f"Predicting on: {args.data_dir}/{args.split}")
    print(f"Data subset: {data_subset}")

    # Build CoT dictionary if cot_path is provided
    cot_dict = None
    if args.cot_path:
        cot_dict, total_cot_entries, missing_cot_count = build_cot_dict(args.cot_path)
        print(f"CoT mode enabled: Using thoughts from {args.cot_path}")
    else:
        print("CoT mode disabled: Using original instructions")

    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=len(DEVICES),initializer=_init_llm,initargs=(args.model_path,)) as poolexec:
        tasks = []
        print("Moving model to devices")
        futures = [poolexec.submit(move_to, dev) for dev in DEVICES]
        for fut in futures: 
            print(fut.result())

        for dataset in data_subset:
            save_dir = os.path.join(args.output_dir, dataset)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            episode_dir = os.path.join(args.data_dir, args.split, dataset)
            output_file = os.path.join(save_dir, "predict.jsonl")

            # Get the list of all episodes files
            if os.path.exists(episode_dir):
                episodes_files = os.listdir(episode_dir)
            else:
                continue

            future = []
            all_tasks = []
            skipped_count = 0
            total_episodes = 0
            
            print("Loading episodes")
            with ThreadPoolExecutor(max_workers=16) as executor:
                for episodes_file in episodes_files:

                    episodes_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}.json")
                    try:
                        with open(episodes_path, 'r', encoding='utf-8') as f:
                            episodes = json.load(f)
                    except Exception as e:
                        print(f"Failed to load {episodes_path}: {e}")
                        continue
                        # Skip this file on error

                    for episode in episodes:
                        total_episodes += 1
                        episode["category"] = dataset
                        image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{episode['step_id']}.jpeg")
                        if not os.path.exists(image_path):
                            image_path = image_path.replace(".jpeg", ".png")
                            if not os.path.exists(image_path):
                                image_path = episode['image_path']
                        future.append(executor.submit(load_image, episode, image_path, args.data_name, cot_dict))

                for f in as_completed(future):
                    result = f.result()
                    if result is not None:
                        all_tasks.append(result)
                    else:
                        skipped_count += 1

            if cot_dict is not None:
                print(f"Skipped {skipped_count} episodes due to missing CoT out of {total_episodes} total episodes ({skipped_count/total_episodes:.2%})")

            with open(output_file, "w", encoding="utf-8") as f_out:
                print("Predicting")
                tasks = []
                for task_value in all_tasks:
                    tasks.append(poolexec.submit(run_episode, *task_value))

                for task in tqdm(as_completed(tasks), total=len(tasks), dynamic_ncols=True):
                    try:
                        episode = task.result()
                        episode_json = json.dumps(episode, ensure_ascii=False)
                        f_out.write(episode_json + "\n")
                        f_out.flush()
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            print(f"Prediction saved at: {output_file}.")
    os.system(f"cat {args.output_dir}/*/predict.jsonl > {args.output_dir}/all.jsonl")
    print(f"Merged prediction saved at: {args.output_dir}/all.jsonl.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GUI Agent Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default="../model/AgentCPM-GUI", help="Model path")
    parser.add_argument("--output_dir", type=str, default="./eval_results/UI-TARS-72B-DPO_cot/android_control_high_test", help="Directory to save results")
    parser.add_argument("--data_name", type=str, default="android_control_high_test", choices=['gui_odyssey_test', 'chinese_app_test', 'aitz_test', 'android_control_high_test', 'android_control_low_test'], help="Eval dataset name")
    parser.add_argument(
        "--cot_path",
        type=str,
        default="/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AndroidControl_high_UI-TARS-72B-DPO_raw.json",
        help="Path to CoT file: either JSONL (one JSON per line, contains pred.thought) or eval JSON with 'detailed_results' and 'predicted_thought'."
    )
    args = parser.parse_args()
    random.seed(args.seed)

    print(f'Loading model at : {args.model_path}')
    print(f'Saving results at: {args.output_dir}')

    predict(args)
    # 测试 build_cot_dict eval/eval_results/AndroidControl_high_UI-TARS-7B-DPO_raw.json
    # build_cot_dict("/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/eval/eval_results/AndroidControl_high_UI-TARS-72B-DPO_raw.json")
    # print(f'Loading model at : ')
