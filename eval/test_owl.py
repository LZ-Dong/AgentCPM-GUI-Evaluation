import base64
from io import BytesIO
from openai import OpenAI
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
import json
from PIL import Image
from utils.qwen_mobile_tool import MobileUse
# from utils.common import pil_to_base64, message_translate, parse_tags, extract_bboxes_from_brackets, draw_point, slim_messages
from IPython.display import display

def build_system_messages(resized_width, resized_height):
    

    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        # TODO
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

    # messages[0]['content'][0]['type'] = 'text'
    # messages[0]['content'][1]['type'] = 'text'

    system_prompt_part = {'role': 'system', 'content': []} # TODO
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

# screenshot = "../assets/screenshot/mobile_example.jpg"
# dummy_image = Image.open(screenshot)


# MIN_PIXELS=3136
# MAX_PIXELS=10035200
# resized_height, resized_width  = smart_resize(dummy_image.height,
#     dummy_image.width,
#     factor=28,
#     min_pixels=MIN_PIXELS,
#     max_pixels=MAX_PIXELS,)
# dummy_image = dummy_image.resize((resized_width, resized_height))

# system_messages = build_system_messages(dummy_image.height, dummy_image.width)
system_messages = build_system_messages(1920, 1080)
user_messages = build_user_messages("View a review of '三文鱼' in '大众点评'", enable_think=True, history=["Open '大众点评' App", "Swipe up to scroll down"])

## private image url usage
# from x.io.oss_info import OssMan
# osm = OssMan()
# user_messages['content'].append({"image": osm.make_tmp_image_url(dummy_image)})

## public image usage
# user_messages['content'].append({"image": f"data:image/png;base64,{pil_to_base64(dummy_image)}"})

messages = [system_messages, user_messages]

print(messages)

# messages_oai = message_translate(messages, to_format='openai')
# print(json.dumps(messages_oai, indent=2, ensure_ascii=False))

# ## Deploy your own vllm openai server
# bot = OpenAI(
#             api_key='fill it',
#             base_url='fill it',
#             timeout=30
#         )
# model_name = 'fill it'

# chat_completion_from_url = bot.chat.completions.create(
#     model=model_name, 
#     messages=messages_oai, **{})
# result = chat_completion_from_url.choices[0].message.content

# action_content = json.loads(parse_tags(result, ['tool_call'])['tool_call'])['arguments']

# print(action_content)
# if 'coordinate' in action_content:
#     dummy_image = draw_point(dummy_image, action_content['coordinate'])
# dummy_image