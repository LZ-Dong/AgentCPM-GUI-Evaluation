# GUI Annotation Tool for GTA/CAA Evaluation
# Only non-trivial sections are commented.

import os, json, argparse, html
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
import gradio as gr

import annotation as anno

# ========== Utility Functions ==========

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def resolve_image_path(example: Dict[str, Any], input_jsonl: str) -> Optional[str]:
    """Try both absolute/relative hints and resolve against JSONL's directory."""
    base = os.path.dirname(input_jsonl)
    cands = []
    for key in ("image_full_path", "image_path"):
        v = example.get(key)
        if v:
            cands += [os.path.join(base, v), v]
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def draw_points(img_path: str, pts: Dict[str, Optional[Tuple[int, int]]]) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    for name, color in [("gt", (0, 255, 0, 200)), ("pred", (0, 128, 255, 200))]:
        p = pts.get(name)
        if isinstance(p, (tuple, list)) and len(p) == 2 and all(isinstance(v, int) for v in p):
            y, x = p
            r = max(20, int(min(im.size) * 0.01))
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color, outline=(0, 0, 0, 120))
    return im


def _mk_info_md(ex: Dict[str, Any], img_path: Optional[str], gt, gt_type, pd, pd_type) -> str:
    instr = ex.get("instruction", "")
    pred_obj = ex.get("pred", {}) or {}
    cot = str(pred_obj.get("thought", "")) if isinstance(pred_obj, dict) else ""

    def esc(x):
        return html.escape(str(x))

    md = ["### Sample Information"]
    if instr:
        md.append(f"**Instruction**: {esc(instr)}")
    if cot:
        md.append(f"**Chain of Thought**: {esc(cot)}")
    md.append(f"**Ground Truth**: `{esc(gt_type)}` · {esc(gt)}")
    md.append(f"**Prediction**: `{esc(pd_type)}` · {esc(pd)}")
    return "\n\n".join(md)


# ========== State Management and View Functions ==========

def init_state(input_path: str):
    """Load data and any existing annotations, index examples by a stable _id.
    - Existing annotations are stored as a map for O(1) lookup and overwrite-safe writes.
    - _id is built from (episode_id, step_id) to make the output file order-agnostic.
    """
    data = load_jsonl(input_path)
    out_path = os.path.join(os.path.dirname(input_path), "annotations.jsonl")

    exists: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    exists[obj["id"]] = obj
                except Exception:
                    pass

    for ex in data:
        ex["_id"] = f'{ex.get("episode_id", "")}:{ex.get("step_id", "")}'

    return dict(data=data, idx=0, fp=input_path, out=out_path, ann=exists)


def get_example_view(state):
    data, idx, fp = state["data"], state["idx"], state["fp"]
    ex = data[idx]
    img_path = resolve_image_path(ex, fp)
    gt, gt_type = anno.extract_gt_action(ex)
    pd, pd_type = anno.extract_pred_action(ex)

    pts = {
        "gt": gt if isinstance(gt, (tuple, list)) else None,
        "pred": pd if isinstance(pd, (tuple, list)) else None,
    }

    if img_path and os.path.exists(img_path):
        vis = draw_points(img_path, pts)
    else:
        vis = Image.new("RGB", (540, 960), (240, 240, 240))

    info_md = _mk_info_md(ex, img_path, gt, gt_type, pd, pd_type)
    return vis, info_md


def _defaults_for_current(state):
    ex = state["data"][state["idx"]]
    a = state["ann"].get(ex["_id"])  # Use saved values if present to avoid clobbering.
    if a:
        return a.get("gta", "NA"), a.get("error_code", "")
    return "NA", ""


def save_annotation(state, gta, err):
    ex = state["data"][state["idx"]]
    ann = {"id": ex["_id"], "gta": gta, "error_code": err}
    state["ann"][ex["_id"]] = ann

    # Atomic-ish: rewrite the small JSONL on every save keeps things simple and consistent.
    with open(state["out"], "w", encoding="utf-8") as f:
        for v in state["ann"].values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    return f"Saved #{state['idx']+1}/{len(state['data'])}"


def goto(state, delta):
    n = len(state["data"])
    prev = state["idx"]
    state["idx"] = max(0, min(n - 1, state["idx"] + delta))
    view = get_example_view(state)
    msg = f"Now: {state['idx']+1}/{n}"
    if state["idx"] == prev and delta > 0:
        msg += " | Already at last item"
    g_def, e_def = _defaults_for_current(state)
    return view, msg, g_def, e_def


def save_and_next(state, gta, err):
    save_msg = save_annotation(state, gta, err)
    (image, info_md), nav_msg, g_def, e_def = goto(state, +1)
    return (f"{save_msg} | {nav_msg}", image, info_md, g_def, e_def)


def jump_to(state, target_one_based):
    """Robust jump: clamp index and tolerate non-int inputs."""
    try:
        t = int(target_one_based) if target_one_based is not None else 1
    except Exception:
        t = 1
    n = len(state["data"])
    t = max(1, min(n, t))
    state["idx"] = t - 1

    image, info_md = get_example_view(state)
    g_def, e_def = _defaults_for_current(state)
    return (f"Jumped: {t}/{n}", image, info_md, g_def, e_def)


# ========== Gradio Application Interface ==========

def build_app(default_input: str):
    """Construct layout with minimal vertical waste; prefill form using saved annotations.
    - The .load() hook initializes state and *outputs default radio/dropdown values* so
      user edits don't get wiped on first render.
    - Navigation ops (save_next/jump_to) always return the form defaults of the new item.
    """
    with gr.Blocks(
        title="GTA/CAA Annotator — Minimal",
        theme="soft",
        css=(
            ".gradio-container {max-width: 1400px !important; margin-left: auto; margin-right: auto;}"
            ".gr-form, .gr-box {padding: 8px 10px;}"
            "* {font-family: Arial, Helvetica, sans-serif !important;}"
        ),
    ) as demo:
        status = gr.Markdown()
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=7, min_width=520):
                gr.Markdown("### Screenshot (Green=GT)")
                img = gr.Image(label=None, interactive=False, height=600)
            with gr.Column(scale=6, min_width=460):
                info = gr.Markdown("", label=None)
                gr.Markdown("**Annotation Form**")
                with gr.Row():
                    gta = gr.Radio(choices=["1", "0", "NA"], value="NA", label="GTA", scale=1)
                err = gr.Dropdown(
                    choices=[
                    "",
                    "E_GT_ERROR",
                    "E_COT_MISSING",
                    "E_COT_PARSE_FAIL",
                    "E_PRED_MISSING",
                    "E_IMAGE_NOT_FOUND",
                    "E_DATA_BAD",
                    "E_MISMATCH_INSTR_SCREEN",
                    "E_OTHER",
                ], value="", label="Error Code"
                )
                btn_save_next = gr.Button("Save and Next →", variant="primary")
                with gr.Row():
                    jump_idx = gr.Number(label="Jump to Index", value=1, precision=0, scale=2)
                    btn_jump = gr.Button("Jump", scale=1)

        def _initial():
            st = init_state(default_input)
            image, info_md = get_example_view(st)
            g_def, e_def = _defaults_for_current(st)
            return (st, image, info_md, f"Loaded {len(st['data'])} steps. Now 1/{len(st['data'])}", g_def, e_def)

        demo.load(_initial, inputs=None, outputs=[state, img, info, status, gta, err])
        btn_save_next.click(save_and_next, [state, gta, err], [status, img, info, gta, err])
        btn_jump.click(jump_to, [state, jump_idx], [status, img, info, gta, err])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file (e.g. sampled_200.jsonl)")
    ap.add_argument("--port", type=int, default=7861, help="Port number for the web interface (default: 7860)")
    ap.add_argument("--share", action="store_true", help="Create a public link")
    args = ap.parse_args()
    app = build_app(args.input)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
    
# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/aitz_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/aitz_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/chinese_app_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/chinese_app_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/chinese_app_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/android_control_high_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/UI-TARS-1.5-7B/android_control_high_test/sampled.jsonl

# python annot_ui_min.py --input /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/GUI-Owl-7B/android_control_high_test/sampled.jsonl