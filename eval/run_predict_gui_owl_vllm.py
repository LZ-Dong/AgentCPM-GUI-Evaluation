from openai import OpenAI
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
import json
from PIL import Image
from utils.utils_owl.mobile_use import MobileUse
from utils.utils_owl.common import pil_to_base64, message_translate, parse_tags, extract_bboxes_from_brackets, draw_point, slim_messages

import sys
import argparse
import copy
import multiprocessing
import os
import threading
import datetime
import fcntl  # For file locking on Linux
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_DEPRECATED_BEAM_SEARCH"]="1"
import random
import jsonschema
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from utils.utils import get_dataset_dir
from utils.action_utils import *

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Add the current file's directory to sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)

ACTION_THOUGHT_SCHEMA = json.load(open(os.path.join(current_dir, 'utils','schema' ,'schema.json'), encoding="utf-8"))

# Configuration constants
class Config:
    API_BASE_URL = "http://localhost:4243/v1"
    MODEL_NAME = "GUI-Owl-32B"
    REQUEST_TIMEOUT = 30
    MIN_PIXELS = 3136
    MAX_PIXELS = 10035200
    RESIZE_FACTOR = 28
    DEFAULT_LONG_PRESS_TIME = 2
    DEFAULT_WAIT_TIME = 2

# DEBUG时将进程数设为1
MAX_WORKERS_PROCESS = 5
MAX_WORKERS_THREAD = 16

def save_raw_output(user_query: str, screenshot: str, history_actions: list, output_text: str, output_dir: str = None):
    """
    Save raw output information to a jsonl file
    
    Args:
        user_query: user query text
        screenshot: screenshot path
        history_actions: list of history actions
        output_text: model output text
        output_dir: output directory path
    """
    try:
        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_query": user_query,
            "screenshot": screenshot,
            "output_text": output_text
        }
        
        # Define output file path
        if output_dir:
            log_file = os.path.join(output_dir, "raw_output.jsonl")
        else:
            log_file = "raw_output.jsonl"
        
        # Thread-safe file writing with file locking
        with open(log_file, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
    except Exception as e:
        print(f"Error saving raw output: {e}")

def aitw_2_qwen2_5_action(aitw_action: dict, resized_height: int, resized_width: int) -> str:
    """
    Convert AITW action to Qwen2.5 action format
    """
    ex_action_type = aitw_action['result_action_type']
    qwen_action = {"name": "mobile_use", "arguments": {}}

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            # Click action
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x* resized_width)
            click_y = int(click_y* resized_height)
            qwen_action["arguments"] = {
                "action": "click",
                "coordinate": [click_x, click_y]
            }
        else:
            # Swipe action
            qwen_action["arguments"] = {
                "action": "swipe",
                "coordinate": [int(touch_yx[1]* resized_width), int(touch_yx[0]* resized_height)],  # Start point
                "coordinate2": [int(lift_yx[1]* resized_width), int(lift_yx[0]* resized_height)]    # End point
            }
    
    elif ex_action_type == ActionType.PRESS_BACK:
        button = "Back"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    
    elif ex_action_type == ActionType.PRESS_HOME:
        button = "Home"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    elif ex_action_type == ActionType.PRESS_ENTER:
        button = "Enter"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    elif ex_action_type == ActionType.TYPE:
        qwen_action["arguments"] = {
            "action": "type",
            "text": aitw_action['result_action_text']
        }
    
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        qwen_action["arguments"] = {
            "action": "terminate",
            "status": "success"
        }
    
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        qwen_action["arguments"] = {
            "action": "terminate",
            "status": "failure"
        }
    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        qwen_action["arguments"] = {
            "action": "long_press",
            "coordinate": [int(touch_yx[1]* resized_width), int(touch_yx[0]* resized_height)],
            "time": Config.DEFAULT_LONG_PRESS_TIME
        }
    elif ex_action_type == ActionType.NO_ACTION:
        qwen_action["arguments"] = {
            "action": "wait",
            "time": Config.DEFAULT_WAIT_TIME
        }
    else:
        print('aitw_action:',aitw_action)
        raise NotImplementedError

    # Return formatted JSON string
    return json.dumps(qwen_action)


def aitw_2_qwen2_5(aitw_action: dict, resized_height: int, resized_width: int) -> str:
    """
    Convert AITW action to Qwen2.5 prompt
    """
    aitw_action = json.loads(aitw_action)
    action=aitw_2_qwen2_5_action(aitw_action,resized_height, resized_width)
    thinking = f"<thinking>\n{aitw_action['coat_action_think']}\n</thinking>\n"
    action = f"<tool_call>\n{action}\n</tool_call>\n"
    result = f'<conclusion>\n"{aitw_action["coat_action_desc"]}"\n</conclusion>'
    return thinking + action + result


def build_system_messages(resized_width, resized_height):
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )
    query_messages = [
        Message(
            role="system", content=[ContentItem(text="You are a helpful assistant.")]
        )
    ]
    messages = NousFnCallPrompt().preprocess_fncall_messages(
        messages=query_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    messages = [m.model_dump() for m in messages]
    system_prompt_part = {'role': 'system', 'content': []}
    system_prompt_part['content'].append(
        {'text': messages[0]['content'][0]['text'] + messages[0]['content'][1]['text']})
    return system_prompt_part


def build_user_messages(instruction, enable_think=False, history=None, think_tag_begin='<thinking>', think_tag_end='</thinking>'):
    if history is None:
        history = []
    user_prompt = f'''The user query: {instruction}'''
    history = ''.join([f'Step {si+1}: {_}; 'for si, _ in enumerate(history)])
    user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
    if enable_think:
        user_prompt += f'\nBefore answering, explain your reasoning step-by-step in {think_tag_begin}{think_tag_end} tags, and insert them before the <tool_call></tool_call> XML tags.'
        user_prompt += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'
    user_messages = {"role": "user", "content": [{"text": user_prompt + '\n'}]}
    return user_messages


def get_qwen_response(user_query: str, screenshot: str, history_actions: list, output_dir: str = None) -> tuple:
    """
    Get the response from the Qwen model
    
    Args:
        user_query: user query text
        screenshot: screenshot path
        
    Returns:
        tuple: (response_text, status_code)
    """
    try:
        # process image size
        dummy_image = Image.open(screenshot)
        resized_height, resized_width = smart_resize(
            dummy_image.height,
            dummy_image.width,
            factor=Config.RESIZE_FACTOR,
            min_pixels=Config.MIN_PIXELS,
            max_pixels=Config.MAX_PIXELS,
        )
        dummy_image = dummy_image.resize((resized_width, resized_height))
        # print(history_actions)
        if history_actions:
            history_actions_list = [aitw_2_qwen2_5_action(action, resized_height, resized_width) for action in history_actions]
        else:
            history_actions_list = None

        # build message
        system_messages = build_system_messages(dummy_image.width, dummy_image.height)
        user_messages = build_user_messages(user_query, enable_think=True, history=history_actions_list)
        user_messages['content'].append({"image": f"data:image/png;base64,{pil_to_base64(dummy_image)}"})
        messages = [system_messages, user_messages]
        messages_oai = message_translate(messages, to_format='openai')
        bot = OpenAI(
            api_key="EMPTY",
            base_url=Config.API_BASE_URL,
            timeout=Config.REQUEST_TIMEOUT,
        )

        chat_completion_from_url = bot.chat.completions.create(
            model=Config.MODEL_NAME, 
            messages=messages_oai, **{})
        output_text = chat_completion_from_url.choices[0].message.content

        # Save raw output information
        save_raw_output(user_query, screenshot, history_actions, output_text, output_dir)

        # print('output_text:',output_text)
        minicpm_answer=qwen2_5_2_minicpm(output_text,resized_height, resized_width)
        # print('minicpm_answer:',json.dumps(minicpm_answer))
        return json.dumps(minicpm_answer), 200
        
    except Exception as e:
        import traceback
        print('error:', str(e))
        print('traceback:', traceback.format_exc())
        return str(e), 500


def qwen2_5_2_minicpm(output_text: str, resized_height: int, resized_width: int) -> dict:
    """
    Convert Qwen2.5's output to minicpm's output
    """
    thought = output_text.split('<thinking>')[1].split('</thinking>')[0].strip() if '<thinking>' in output_text and '</thinking>' in output_text else ""
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    qwen_action = action['arguments']
    action_name = qwen_action['action']
    # handle click action, long_press is directly processed as click because there is no corresponding action
    if action_name == "click" :
        x, y = qwen_action["coordinate"]

        # normalize
        x = x/ resized_width*1000
        y = y/ resized_height*1000
        return {"thought": thought, "POINT": [int(x), int(y)]}
    elif action_name == "long_press":
        x, y = qwen_action["coordinate"]
        x = x/ resized_width*1000
        y = y/ resized_height*1000
        try:
            time=qwen_action["time"]
        except:
            time=Config.DEFAULT_LONG_PRESS_TIME
        # convert time to milliseconds
        time = time*1000
        return {"thought": thought, "POINT": [int(x), int(y)], "duration": time}
    
    # handle swipe action
    elif action_name == "swipe":
        x1, y1 = qwen_action["coordinate"]
        x2, y2 = qwen_action["coordinate2"]
        x1 = x1/ resized_width*1000
        y1 = y1/ resized_height*1000
        x2 = x2/ resized_width*1000
        y2 = y2/ resized_height*1000
        # determine swipe direction based on start and end points
        if abs(x2 - x1) > abs(y2 - y1):  # horizontal swipe
            direction = "right" if x2 > x1 else "left"
        else:  # vertical swipe
            direction = "down" if y2 > y1 else "up"
        return {"thought": thought, "POINT": [int(x1), int(y1)], "to": direction}
    
    # handle input text
    elif action_name == "type":
        return {"thought": thought, "TYPE": qwen_action["text"]}
    
    # handle system button
    elif action_name == "system_button":
        button = qwen_action["button"]
        if button == "Back":
            return {"thought": thought, "PRESS": "BACK"}
        elif button == "Home":
            return {"thought": thought, "PRESS": "HOME"}
        elif button == "Enter":
            return {"thought": thought, "PRESS": "ENTER"}

    # handle terminate action
    elif action_name == "terminate":
        return {"thought": thought, "STATUS": "finish"}
    elif action_name == "wait":
        # convert time to milliseconds
        time = qwen_action["time"]
        time = time*1000    
        return {"thought": thought, "duration": time}
    
    # for other actions (such as key,open, etc.), they may need to be ignored or specially processed
    #key wait cannot find corresponding action
    return {}


def run_episode(episode, image_path, history_list, output_dir):
    query = episode["instruction"]
    screenshot = image_path
    output_text, status_code = get_qwen_response(query, screenshot, history_list, output_dir)
    episode["pred"] = extract_and_validate_json(output_text)
    return episode


def extract_and_validate_json(input_string):
    # if input_string == "":
        # raise ValueError("Error, empty output.")
    try:
        json_obj = json.loads(input_string)
        # validate JSON data against Schema
        jsonschema.validate(json_obj, ACTION_THOUGHT_SCHEMA)
        return json_obj
    except json.JSONDecodeError as e:
        print("Error, JSON is NOT valid.",input_string,"over")
        return input_string
    except Exception as e:
        print("Error, JSON is NOT valid according to the schema.",input_string,"over")
        return input_string


def load_image(episode, image_path, history_list):
    return (episode, image_path, history_list)


def predict(args, datasets):
    data_dir = args.data_dir
    split_type = args.split
    print("Predicting on:",datasets)
    
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS_PROCESS) as poolexec:
        for dataset in datasets:
            save_dir = os.path.join(args.output_dir, dataset)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            episode_dir = os.path.join(data_dir, split_type, dataset)

            # Use predict.jsonl file to store results (write line by line)
            output_file = os.path.join(save_dir, "predict.jsonl")
            
            # Get the list of all episodes files
            if os.path.exists(episode_dir):
                episodes_files = os.listdir(episode_dir)
            else:
                continue
            
            future = []
            all_tasks = []
            print("Loading episodes")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_THREAD) as executor:
                for episodes_file in episodes_files:

                    episodes_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}.json")
                    try:
                        with open(episodes_path, 'r', encoding='utf-8') as f:
                            episodes = json.load(f)
                    except Exception as e:
                        print(f"Failed to load {episodes_path}: {e}")
                        continue
                        # Skip this file on error
                    for index,episode in enumerate(episodes):
                        episode_history = []  # Create a separate history for each episode
                        for prev_episode in episodes[:index]:
                        #for prev_episode in episodes[:episode['step_id']-1]:  # Only get history before current step
                            image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{prev_episode['step_id']}.jpeg")
                            if not os.path.exists(image_path):
                                image_path = image_path.replace(".jpeg", ".png")
                            if not os.path.exists(image_path):
                                image_path = prev_episode["image_path"]
                            histroy_action = {
                                "result_action_type": prev_episode['result_action_type'],
                                "result_action_text": prev_episode['result_action_text'],
                                "result_touch_yx": prev_episode['result_touch_yx'],
                                "result_lift_yx": prev_episode['result_lift_yx'],
                                "low_instruction": prev_episode.get('low_instruction', ''),
                                "image_path": image_path,
                                "result_action_app_name": prev_episode.get('result_action_app_name', ''),
                            }
                            episode_history.append(histroy_action)
                        episode["category"] = dataset
                        image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{episode['step_id']}.jpeg")
                        if not os.path.exists(image_path):
                            image_path = image_path.replace(".jpeg", ".png")
                        if not os.path.exists(image_path):
                            image_path = episode["image_path"]
                        episode_copy = copy.deepcopy(episode)
                        episode_history_copy = copy.deepcopy(episode_history)
                        future.append(executor.submit(load_image, episode_copy, image_path, episode_history_copy))

                for f in as_completed(future):
                    all_tasks.append(f.result())

            with open(output_file, "w", encoding="utf-8") as f_out:
                print("Predicting")
                tasks = []
                for task_value in all_tasks:
                    tasks.append(poolexec.submit(run_episode, *task_value, args.output_dir))
                
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
    parser = argparse.ArgumentParser(description="Qwen2.5VL Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    # parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/share_data/data1/GUI_eval/Qwen2.5-VL-7B-Instruct"),
    #                    help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.getenv('OUTPUT_PATH', "./eval_results/GUI-Owl-7B/chinese_app_test")),
                       help="Directory to save results")
    parser.add_argument("--data_name", type=str, default=os.getenv("PREDICT_DATASET", "chinese_app_test"),
                       help="Eval dataset name")
    args = parser.parse_args()
    random.seed(args.seed)

    # Get dataset information
    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    
    # Update output directory with model name
    # model_name = args.model_path.split("/")[-2:]  # Get last two parts of model path
    # args.output_dir = os.path.join(args.output_dir, *model_name, args.data_name)
    
    # print(f'Loading model at : {args.model_path}')
    print(f'Loading data at  : {args.data_dir}')
    print(f'Processing subsets: {data_subset}')
    print(f'Saving results at: {args.output_dir}')
    
    predict(args, data_subset)
