"""
Minimal Human Annotation UI for GTA/CAA

- Loads tasks from a JSONL file (one step per line)
- Shows instruction + screenshot
- Overlays GT point (green), Model point (blue), CoT-implied point (red)
- Click on image to set the CoT-implied point
- Fill form: implied action type/text, ambiguity flags, GTA/CAA, confidence, rationale, annotator id
- Save per-step annotation; navigate between steps; export JSONL

{
    "episode_id": "8483176786914864439",
    "episode_length": 7,
    "step_id": 0,
    "instruction": "What is the capital of Germany?",
    "ui_positions": "[[0.1, 0.18518518518518517, 0.027777777777777776, 0.39814814814814814], [0.10092592592592593, 0.6592592592592592, 0.020370370370370372, 0.1111111111111111], [0.7592592592592593, 0.46296296296296297, 0.058333333333333334, 0.07407407407407407], [0.7592592592592593, 0.65, 0.06666666666666667, 0.2851851851851852], [0.7675925925925926, 0.2777777777777778, 0.03981481481481482, 0.05740740740740741], [0.7777777777777778, 0.11481481481481481, 0.022222222222222223, 0.024074074074074074], [0.862037037037037, 0.10185185185185185, 0.03981481481481482, 0.044444444444444446], [0.950925925925926, 0.19444444444444445, 0.030555555555555555, 0.03148148148148148], [0.950925925925926, 0.47962962962962963, 0.028703703703703703, 0.03333333333333333], [0.950925925925926, 0.7666666666666667, 0.028703703703703703, 0.03333333333333333]]",
    "ui_text": "[\"Monday, Oct 10\", \"58\\u00b0F\", \"\", \"90\", \"\", \"\", \"G\", \"\", \"\", \"\"]",
    "ui_types": "[\"TEXT\", \"TEXT\", \"ICON_TIME\", \"TEXT\", \"ICON_CHAT\", \"ICON_PLAY\", \"ICON_GOOGLE\", \"ICON_V_BACKWARD\", \"ICON_NAV_BAR_CIRCLE\", \"ICON_NAV_BAR_RECT\"]",
    "result_action_type": 4,
    "result_action_text": "",
    "result_touch_yx": "[0.7861145734786987, 0.7045074105262756]",
    "result_lift_yx": "[0.7861145734786987, 0.7045074105262756]",
    "image_path": "general/GENERAL-8483176786914864439/GENERAL-8483176786914864439_0.png",
    "image_full_path": "android-in-the-wild/aitw_with_gpt/test/general/GENERAL-8483176786914864439/GENERAL-8483176786914864439_0.png",
    "coat_screen_desc": "This is a screenshot of a smartphone home screen displaying a clean and simple layout. The top shows the status bar with time, Bluetooth, signal, and battery indicators; it also features the date \"Monday, Oct 10\" and the temperature \"58°F.\" At the bottom, there is a search bar widget alongside a row of app icons including YouTube, Messages, Phone, Chrome, and Photos, which are commonly used applications. Below these icons, there is a navigation bar with back, home, and recent apps buttons for navigating the phone's interface. The background is a gradient from pink to deep purple, providing a visually appealing aesthetic.",
    "coat_action_think": "To address the query, the Google Chrome app can be used to search for the capital of Germany, or the Google search bar widget at the bottom of the screen can directly be utilized to type in the query and get the result. Possible actions are to open the Chrome browser or use the Google search bar widget to find the information.",
    "coat_action_desc": "click on the Chrome browser icon located at the lower right part of the screen.",
    "coat_action_result": "By doing so, the Chrome browser application is opened, providing access to the internet where information on the capital of Germany can be searched. This enables the user to quickly find that Berlin is the capital of Germany.",
    "image_height": 1080,
    "image_width": 540,
    "image_channels": 3,
    "category": "general",
    "pred": {
        "thought": "The Google search bar on the home screen is relevant for answering the query about the capital of Germany. Possible actions are to tap on the Google search bar and type in the query \"What is the capital of Germany?\" to obtain the answer.\nclick on the Google Search Bar located at the bottom right.",
        "POINT": [
            689,
            790
        ]
    }
}
"""
import json
from utils.action_type import ActionType
import numpy as np
import jax.numpy as jnp
# from utils.action_utils import extract_gt_action
from utils.convert_output import parse_action
import logging
# 屏蔽jax的warning
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

'''======================================
            解析Ground Truth Action
   ======================================'''
# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04

def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = jnp.linalg.norm(
        jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def extract_gt_action(example):
    ex_action_type = example['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(example['result_lift_yx'])
        touch_yx = json.loads(example['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            action_type = 'CLICK'
            w, h = example['image_width'], example['image_height']
            assert w and h, "Invalid image size"
            click_y, click_x = round(lift_yx[0] * h), round(lift_yx[1] * w) # Note: image size range, not 0-1000
            action = (click_y, click_x)
        else:
            action_type = 'SCROLL'
            v_change = abs(touch_yx[0] - lift_yx[0])
            h_change = abs(lift_yx[1] - touch_yx[1])
            # 手指向上滑动 -> 内容向下滚动；手指向下滑动 -> 内容向上滚动
            is_scroll_down = lift_yx[0] < touch_yx[0]     # lift位置更靠上，手指向上滑
            # 手指向左滑动 -> 内容向右滚动；手指向右滑动 -> 内容向左滚动  
            is_scroll_right = lift_yx[1] < touch_yx[1]   # lift位置更靠左，手指向左滑
            if v_change >= 0.9*h_change:   # vertical
                action = "scroll down" if is_scroll_down else "scroll up"
            else:                          # horizontal
                action = "scroll right" if is_scroll_right else "scroll left"
    elif ex_action_type in (
        ActionType.PRESS_BACK, 
        ActionType.PRESS_HOME,
        ActionType.PRESS_ENTER
    ):  
        button = ActionType(ex_action_type).name.split('_')[1].lower()
        action = f'press {button}'
        action_type = 'PRESS'
    elif ex_action_type == ActionType.TYPE:
        action_text = example['result_action_text']
        action = f'input text "{action_text}"'
        action_type = 'INPUT'
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        action = 'stop and set the query as completed'
        action_type = 'STOP'
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        action = 'stop and set the query as impossible'
        action_type = 'STOP'
    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(example['result_lift_yx'])
        w, h = example['image_width'], example['image_height']
        assert w and h, "Invalid image size"
        click_y, click_x = round(lift_yx[0] * h), round(lift_yx[1] * w) # Note: image size range, not 0-1000
        action = (click_y, click_x)
        action_type = "LONG_POINT"
    elif ex_action_type == ActionType.NO_ACTION:
        duration_time = example['duration']
        action = f'no action for {duration_time} ms'
        action_type = 'NO_ACTION'
    else:
        raise NotImplementedError
    
    return action, action_type

'''======================================
            解析Predicted Truth Action
   ======================================'''
default_duration = 200  # ms

def extract_pred_action(example):
    _action, _args, _status = parse_action(example['pred'])
    _duration = _args.get('duration', default_duration) if _args else None
    _stop_status = [
        "finish",
        "satisfied",
        "impossible",
        "interrupt",
        "need_feedback"
    ]
    if _action is None and _args is None and _status is None:
        print('Schema error.')
        return None, None
    elif _status in _stop_status:
        action = 'stop and set the query as completed'
        action_type = 'STOP'
    elif "TYPE" in _action:
        action = f'input text "{_action["TYPE"]}"'
        action_type = 'INPUT'
    elif "POINT" in _action and "to" not in _args and _duration == default_duration: # click
        w, h = example['image_width'], example['image_height']
        assert w and h, "Invalid image size"
        click_y, click_x = round(_action['POINT'][1] * h / 1000), round(_action['POINT'][0] * w / 1000) # Note: image size range, not 0-1000
        action = (click_y, click_x)
        action_type = 'CLICK'
    elif "POINT" in _action and "to" in _args and _duration == default_duration:
        action = f'scroll {_args["to"]}'
        action_type = 'SCROLL'
    elif "POINT" in _action and "duration" in _args and _duration > default_duration: # long press
        w, h = example['image_width'], example['image_height']
        assert w and h, "Invalid image size"
        click_y, click_x = round(_action['POINT'][1] * h / 1000), round(_action['POINT'][0] * w / 1000) # Note: image size range, not 0-1000
        action = (click_y, click_x)
        action_type = 'LONG_POINT'
    elif "PRESS" in _action:
        action = f'press {_action["PRESS"].lower()}'
        action_type = 'PRESS'
    elif "duration" in _args: # pause and wait
        action = f'no action for {_args["duration"]} ms'
        action_type = 'NO_ACTION'
    else:
        raise ValueError("Unknown action type.")
    
    return action, action_type

def _load_demo_data():
    """仅在直接运行 annotation.py 时演示；被 import 时不触发。"""
    try:
        with open('/data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/all.jsonl', 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]
    except Exception as e:
        print(f"[annotation.py] demo 数据加载失败：{e}")
        return []

if __name__ == "__main__":
    # 仅用于本文件的演示打印；被其它程序 import 时不会执行
    data = _load_demo_data()
    if data:
        for i in range(min(10, len(data))):
            example = data[i]
            print(f"Example {i}:")
            try:
                print("GT Action:", extract_gt_action(example))
                print("Pred Action:", extract_pred_action(example))
                # 尝试多个候选字段
                cot_text = (example.get("pred", {}) or {}).get("thought", "") or example.get("coat_action_think", "")
                print("CoT:", cot_text if cot_text else "(empty)")
            except Exception as e:
                print(f"[annotation.py] demo 打印失败: {e}")

"""
Example 0:
GT Action: ((72, 232), 'CLICK')
Pred Action: ((70, 436), 'CLICK')
CoT: The screen shows results for pizza restaurants, which are irrelevant to the query. Possible actions are to tap on the search bar at the top of the screen and enter "top rated sushi restaurants" to conduct a new search.
click on the search bar located at the top right.
Example 1:
GT Action: ('press enter', 'PRESS ENTER')
Pred Action: ((234, 198), 'CLICK')
CoT: The screen shows that the Google search bar is in use but no search results are displayed yet. Possible actions are to select one of the autocomplete suggestions or press enter to execute the search for hotels in London.
click on the "london uk" search suggestion located at middle upper part of the screen.
Example 2:
GT Action: ('stop and set the query as completed', 'STOP')
Pred Action: ('scroll up', 'SCROLL')
CoT: The screen shows a hotel booking interface with options to browse by star rating, which suggests the user can further refine the search results. Possible actions are to continue browsing through different hotel options or star ratings, or if the desired hotel has been found, stop and set the query as completed.
stop and set the query as completed
Example 3:
GT Action: ('scroll up', 'SCROLL')
Pred Action: ((929, 372), 'CLICK')
CoT: The Google Chrome app is visible on the home screen, which can be used to search for the latest news. Possible actions are opening the Google Chrome app and using it to search for the current news stories in the afternoon.
click on the Google Search Bar located at the lower right part of the screen.
Example 4:
GT Action: ('stop and set the query as completed', 'STOP')
Pred Action: ('stop and set the query as completed', 'STOP')
CoT: The screen shows the lock screen with a previous search for "YouTube 2018," which suggests that the YouTube app is not immediately accessible from this screen. Possible actions are to either unlock the phone to access the apps drawer or to use the search bar at the bottom to navigate directly to the YouTube app if it's installed.
press the home button
Example 5:
GT Action: ((953, 214), 'CLICK')
Pred Action: ((946, 226), 'CLICK')
CoT: The screen shows the home screen with common applications but does not display YouTube or any related shortcut. Possible actions are to tap on the Google Chrome icon to use the browser for searching and playing the new Katy Perry video on YouTube, or to look for a YouTube app by swiping up or tapping on the Google search bar to type in "Katy Perry new video" and find it through a search.
click on the Google Search Bar located at the bottom right.
Example 6:
GT Action: ('input text "Search for hotels in London"', 'INPUT')
Pred Action: ('input text "hotels in London"', 'INPUT')
CoT: The screen shows the Google search bar, which can be used to enter a new search query. Possible actions are to tap on the Google search bar and type "hotels in London" to proceed with the hotel search in London as requested by the user.
type in the content: "hotels in London"
Example 7:
GT Action: ((74, 369), 'CLICK')
Pred Action: ((75, 302), 'CLICK')
CoT: The screen shows search results for Panama, which is not related to the current query about news. Possible actions are to tap on the browser's address bar or use the "new tab" icon to start a new search for the latest afternoon news.
click on the search bar located at the top right.
Example 8:
GT Action: ('press home', 'PRESS HOME')
Pred Action: ('press home', 'PRESS HOME')
CoT: The screen shows a contacts application which is not useful for conducting a real estate search. Possible actions are to exit the contacts application and open a web browser or a real estate application where I could search for rental prices for 3 bedroom apartments in Miami.
press the home button
Example 9:
GT Action: ('stop and set the query as completed', 'STOP')
Pred Action: ((240, 211), 'CLICK')
CoT: The screen shows search results for "news" with links to various news organizations and a specific article about PBS NewsHour. Possible actions are to click on the article link or refine the search query to specifically look for afternoon news updates.
click on the search result located at upper middle left.
"""