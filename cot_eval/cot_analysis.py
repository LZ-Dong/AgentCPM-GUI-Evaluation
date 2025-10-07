import json
from typing import Dict
import argparse

def load_em_map_json(result_json_path: str) -> Dict[str, int]:
    em_map: Dict[str, int] = {}
    with open(result_json_path, 'r', encoding='utf-8') as f:
        head = f.read(8192)
        f.seek(0)
        if head.lstrip().startswith('{') and '"detailed_results"' in head:
            data = json.load(f)
            results = data.get('detailed_results', [])
            for item in results:
                eid = item.get('episode_id')
                sid = item.get('step_id') - 1
                em = item.get('is_success', False)
                em_map[f"{eid}:{sid}"] = 1 if em else 0
    return em_map

def load_em_map(result_json_path: str) -> Dict[str, int]:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    em_map: Dict[str, int] = {}
    for item in data:
        eid = str(item.get("episode_id", ""))
        sid = item.get("step_id", None)
        if eid == "" or sid is None:
            continue
        key = f"{eid}:{sid}"
        em_map[key] = 1 if bool(item.get("exact_match", False)) else 0
    return em_map

def quadrants(em: int, gta: int) -> Dict[str, int]:
    return {
        "q1": 1 if (em == 1 and gta == 1) else 0,
        "q2": 1 if (em == 0 and gta == 1) else 0,
        "q3": 1 if (em == 0 and gta == 0) else 0,
        "q4": 1 if (em == 1 and gta == 0) else 0,
    }

def cot_analysis_json(em_json, gta_jsonl, save_path):
    with open(gta_jsonl, 'r') as f:
        gta_data = json.load(f)


    em_map = load_em_map_json(em_json)
    em_sum, gta_sum = 0, 0
    q1, q2, q3, q4 = 0, 0, 0, 0

    for item in gta_data:
        eid = str(item.get("episode_id", ""))
        sid = item.get("step_id", None)
        if eid == "" or sid is None:
            continue
        key = f"{eid}:{sid}"
        gta = 1 if bool(item.get("exact_match", False)) else 0
        em = em_map.get(key)
        if (em == 1 and gta == 1):
            q1 += 1
            em_sum += 1
            gta_sum += 1
        elif (em == 0 and gta == 1):
            q2 += 1
            gta_sum += 1
        elif (em == 0 and gta == 0):
            q3 += 1
        elif (em == 1 and gta == 0):
            q4 += 1
            em_sum += 1

    result = {
        # "total": len(em_data),
        "cot_total": len(gta_data),
        # "missing_rate": missing_rate,
        "em": em_sum/len(gta_data)*100,
        "gta": gta_sum/len(gta_data)*100,
        "q1": q1/len(gta_data)*100,
        "q2": q2/len(gta_data)*100,
        "q3": q3/len(gta_data)*100,
        "q4": q4/len(gta_data)*100,
    }

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    return result

def cot_analysis(em_jsonl, gta_jsonl, save_path):
    with open(em_jsonl, 'r') as f:
        em_data = json.load(f)
    with open(gta_jsonl, 'r') as f:
        gta_data = json.load(f)

    missing_rate = (len(em_data) - len(gta_data)) / len(em_data) * 100

    em_map = load_em_map(em_jsonl)
    em_sum, gta_sum = 0, 0
    q1, q2, q3, q4 = 0, 0, 0, 0

    for item in gta_data:
        eid = str(item.get("episode_id", ""))
        sid = item.get("step_id", None)
        if eid == "" or sid is None:
            continue
        key = f"{eid}:{sid}"
        gta = 1 if bool(item.get("exact_match", False)) else 0
        em = em_map.get(key)
        if (em == 1 and gta == 1):
            q1 += 1
            em_sum += 1
            gta_sum += 1
        elif (em == 0 and gta == 1):
            q2 += 1
            gta_sum += 1
        elif (em == 0 and gta == 0):
            q3 += 1
        elif (em == 1 and gta == 0):
            q4 += 1
            em_sum += 1

    result = {
        "total": len(em_data),
        "cot_total": len(gta_data),
        "missing_rate": missing_rate,
        "em": em_sum/len(gta_data)*100,
        "gta": gta_sum/len(gta_data)*100,
        "q1": q1/len(gta_data)*100,
        "q2": q2/len(gta_data)*100,
        "q3": q3/len(gta_data)*100,
        "q4": q4/len(gta_data)*100,
    }

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    return result

if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--em_jsonl", type=str, required=True, help="Path to the EM result JSON file")
    # argparser.add_argument("--gta_jsonl", type=str, required=True, help="Path to the GTA result JSON file")
    # argparser.add_argument("--save_path", type=str, required=True, help="Path to save the analysis result JSON file")
    # args = argparser.parse_args()
    # result = cot_analysis(args.em_jsonl, args.gta_jsonl, args.save_path)
    # print(result)
    # print(f"Cot analysis saved to {args.save_path}")
    result = cot_analysis_json(
        em_json="eval/eval_results/AndroidControl_high_UI-TARS-2B-SFT_raw.json",
        gta_jsonl="eval/eval_results/UI-TARS-2B-SFT_cot/android_control_high_test/results/result.json",
        save_path="eval/eval_results/UI-TARS-2B-SFT_cot/android_control_high_test/cot_analysis.json"
    )
    print(result)
    # result = cot_analysis_json(
    #     em_json="eval/eval_results/AndroidControl_high_UI-TARS-7B-SFT_raw.json",
    #     gta_jsonl="eval/eval_results/UI-TARS-7B-SFT_cot/android_control_high_test/results/result.json",
    #     save_path="eval/eval_results/UI-TARS-7B-SFT_cot/android_control_high_test/cot_analysis.json"
    # )
    # print(result)
    # result = cot_analysis_json(
    #     em_json="eval/eval_results/AndroidControl_high_UI-TARS-72B-SFT_raw.json",
    #     gta_jsonl="eval/eval_results/UI-TARS-72B-SFT_cot/android_control_high_test/results/result.json",
    #     save_path="eval/eval_results/UI-TARS-72B-SFT_cot/android_control_high_test/cot_analysis.json"
    # )
    # print(result)
    # result = cot_analysis_json(
    #     em_json="eval/eval_results/AndroidControl_high_UI-TARS-7B-DPO_raw.json",
    #     gta_jsonl="eval/eval_results/UI-TARS-7B-DPO_cot/android_control_high_test/results/result.json",
    #     save_path="eval/eval_results/UI-TARS-7B-DPO_cot/android_control_high_test/cot_analysis.json"
    # )
    # print(result)
    # result = cot_analysis_json(
    #     em_json="eval/eval_results/AndroidControl_high_UI-TARS-72B-DPO_raw.json",
    #     gta_jsonl="eval/eval_results/UI-TARS-72B-DPO_cot/android_control_high_test/results/result.json",
    #     save_path="eval/eval_results/UI-TARS-72B-DPO_cot/android_control_high_test/cot_analysis.json"
    # )
    # print(result)

"""Example usage:
# AgentCPM-GUI/android_control_high_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/AgentCPM-GUI/android_control_high_test/results/result.json \
--gta_jsonl eval/eval_results/AgentCPM-GUI_cot/android_control_high_test/results/result.json \
--save_path eval/eval_results/AgentCPM-GUI_cot/android_control_high_test/cot_analysis.json

# AgentCPM-GUI/aitz_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/AgentCPM-GUI/aitz_test/results/result.json \
--gta_jsonl eval/eval_results/AgentCPM-GUI_cot/aitz_test/results/result.json \
--save_path eval/eval_results/AgentCPM-GUI_cot/aitz_test/cot_analysis.json

# AgentCPM-GUI/chinese_app_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/AgentCPM-GUI/chinese_app_test/results/result.json \
--gta_jsonl eval/eval_results/AgentCPM-GUI_cot/chinese_app_test/results/result.json \
--save_path eval/eval_results/AgentCPM-GUI_cot/chinese_app_test/cot_analysis.json

# UI-TARS-1.5-7B/android_control_high_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-1.5-7B/android_control_high_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-1.5-7B_cot/android_control_high_test/results/result.json \
--save_path eval/eval_results/UI-TARS-1.5-7B_cot/android_control_high_test/cot_analysis.json

# UI-TARS-1.5-7B/aitz_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-1.5-7B/aitz_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-1.5-7B_cot/aitz_test/results/result.json \
--save_path eval/eval_results/UI-TARS-1.5-7B_cot/aitz_test/cot_analysis.json

# UI-TARS-1.5-7B/chinese_app_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-1.5-7B/chinese_app_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-1.5-7B_cot/chinese_app_test/results/result.json \
--save_path eval/eval_results/UI-TARS-1.5-7B_cot/chinese_app_test/cot_analysis.json

# GUI-Owl-7B/android_control_high_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/GUI-Owl-7B/android_control_high_test/results/result.json \
--gta_jsonl eval/eval_results/GUI-Owl-7B_cot/android_control_high_test/results/result.json \
--save_path eval/eval_results/GUI-Owl-7B_cot/android_control_high_test/cot_analysis.json

# GUI-Owl-7B/aitz_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/GUI-Owl-7B/aitz_test/results/result.json \
--gta_jsonl eval/eval_results/GUI-Owl-7B_cot/aitz_test/results/result.json \
--save_path eval/eval_results/GUI-Owl-7B_cot/aitz_test/cot_analysis.json

# GUI-Owl-7B/chinese_app_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/GUI-Owl-7B/chinese_app_test/results/result.json \
--gta_jsonl eval/eval_results/GUI-Owl-7B_cot/chinese_app_test/results/result.json \
--save_path eval/eval_results/GUI-Owl-7B_cot/chinese_app_test/cot_analysis.json

# GUI-Owl-32B/android_control_high_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/GUI-Owl-32B/android_control_high_test/results/result.json \
--gta_jsonl eval/eval_results/GUI-Owl-32B_cot/android_control_high_test/results/result.json \
--save_path eval/eval_results/GUI-Owl-32B_cot/android_control_high_test/cot_analysis.json

# GUI-Owl-32B/aitz_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/GUI-Owl-32B/aitz_test/results/result.json \
--gta_jsonl eval/eval_results/GUI-Owl-32B_cot/aitz_test/results/result.json \
--save_path eval/eval_results/GUI-Owl-32B_cot/aitz_test/cot_analysis.json

# GUI-Owl-32B/chinese_app_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/GUI-Owl-32B/chinese_app_test/results/result.json \
--gta_jsonl eval/eval_results/GUI-Owl-32B_cot/chinese_app_test/results/result.json \
--save_path eval/eval_results/GUI-Owl-32B_cot/chinese_app_test/cot_analysis.json

# UI-TARS-7B-SFT/aitz_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-7B-SFT/aitz_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-7B-SFT_cot/aitz_test/results/result.json \
--save_path eval/eval_results/UI-TARS-7B-SFT_cot/aitz_test/cot_analysis.json

# UI-TARS-7B-SFT/chinese_app_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-7B-SFT/chinese_app_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-7B-SFT_cot/chinese_app_test/results/result.json \
--save_path eval/eval_results/UI-TARS-7B-SFT_cot/chinese_app_test/cot_analysis.json

# UI-TARS-7B-SFT/android_control_high_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-7B-SFT/android_control_high_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-7B-SFT_cot/android_control_high_test/results/result.json \
--save_path eval/eval_results/UI-TARS-7B-SFT_cot/android_control_high_test/cot_analysis.json

# UI-TARS-7B-DPO/aitz_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-7B-DPO/aitz_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-7B-DPO_cot/aitz_test/results/result.json \
--save_path eval/eval_results/UI-TARS-7B-DPO_cot/aitz_test/cot_analysis.json

# UI-TARS-7B-DPO/chinese_app_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-7B-DPO/chinese_app_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-7B-DPO_cot/chinese_app_test/results/result.json \
--save_path eval/eval_results/UI-TARS-7B-DPO_cot/chinese_app_test/cot_analysis.json

# UI-TARS-7B-DPO/android_control_high_test
python cot_eval/cot_analysis.py \
--em_jsonl eval/eval_results/UI-TARS-7B-DPO/android_control_high_test/results/result.json \
--gta_jsonl eval/eval_results/UI-TARS-7B-DPO_cot/android_control_high_test/results/result.json \
--save_path eval/eval_results/UI-TARS-7B-DPO_cot/android_control_high_test/cot_analysis.json
"""