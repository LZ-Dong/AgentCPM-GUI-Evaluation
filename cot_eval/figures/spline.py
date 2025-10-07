# -*- coding: utf-8 -*-
# Spline plots: GTA, EM, IDEAL(=EM−RG) across datasets with shared legend and red annotations.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 可选：若环境有 SciPy，则用样条平滑；否则回退为折线
try:
    from scipy.interpolate import make_interp_spline
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ===== Global font scale: 放大到原来的 1.5 倍 =====
plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 1.5})

# ===== 数据（Dataset, Model, EM, GTA, RG）——来自表格 =====
rows = [
    ("AITZ", "AgentCPM-GUI-8B", 76.29, 70.43, 9.38),
    ("AITZ", "UI-TARS-7B-SFT", 66.69, 64.98, 9.44),
    ("AITZ", "UI-TARS-7B-DPO", 65.95, 63.86, 10.57),
    ("AITZ", "UI-TARS-1.5-7B", 58.94, 66.09, 4.69),
    ("AITZ", "GUI-Owl-7B", 61.01, 66.17, 5.12),
    ("AITZ", "GUI-Owl-32B", 59.65, 66.09, 5.76),
    ("CAGUI", "AgentCPM-GUI-8B", 91.21, 85.35, 8.13),
    ("CAGUI", "UI-TARS-7B-SFT", 71.36, 77.86, 2.57),
    ("CAGUI", "UI-TARS-7B-DPO", 70.60, 78.52, 2.44),
    ("CAGUI", "UI-TARS-1.5-7B", 68.42, 81.68, 2.06),
    ("CAGUI", "GUI-Owl-7B", 61.00, 79.87, 2.88),
    ("CAGUI", "GUI-Owl-32B", 65.88, 81.53, 2.64),
    ("AndroidControl", "AgentCPM-GUI-8B", 69.17, 74.45, 3.41),
    ("AndroidControl", "UI-TARS-7B-SFT", 74.71, 81.44, 2.45),
    ("AndroidControl", "UI-TARS-7B-DPO", 73.87, 80.16, 2.85),
    ("AndroidControl", "UI-TARS-1.5-7B", 76.10, 76.22, 5.41),
    ("AndroidControl", "GUI-Owl-7B", 65.22, 72.79, 4.74),
    ("AndroidControl", "GUI-Owl-32B", 69.89, 74.15, 6.18),
]
df = pd.DataFrame(rows, columns=["Dataset","Model","EM","GTA","RG"])
df["IDEAL"] = df["EM"] - df["RG"]  # IDEAL := EM − RG

# 保持固定的数据集顺序
datasets = ["AITZ", "CAGUI", "AndroidControl"]

# ===== 绘制三个并列子图，共享 y 轴 =====
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

# 需要的曲线及显示顺序（用于共享图例）
order_labels = ["GTA", "EM", "IDEAL"]
colors = {"GTA": "tab:green", "EM": "tab:blue", "IDEAL": "tab:orange"}

# 用于收集共享图例的句柄
line_handles = {}

for ax, dataset in zip(axes, datasets):
    sub = df[df["Dataset"] == dataset].copy().reset_index(drop=True)
    x = np.arange(len(sub))
    xnew = np.linspace(x.min(), x.max(), 200)

    # 依次绘制 GTA、EM、IDEAL（确保共享图例顺序一致）
    for key in order_labels:
        y = sub[key].values
        if HAS_SCIPY and len(sub) >= 4:  # 充足点数时做三次样条
            try:
                spl = make_interp_spline(x, y, k=3)
                ynew = spl(xnew)
                ln, = ax.plot(xnew, ynew, linewidth=2, label=key, color=colors[key])
            except Exception:
                ln, = ax.plot(x, y, linewidth=2, label=key, color=colors[key])
        else:
            ln, = ax.plot(x, y, linewidth=2, label=key, color=colors[key])

        ax.scatter(x, y, color=colors[key], zorder=3)
        # 只记录一次句柄用于共享图例
        if key not in line_handles:
            line_handles[key] = ln

    ax.set_xticks(x)
    ax.set_xticklabels(sub["Model"], rotation=45, ha="right")
    ax.set_title(dataset)
    ax.set_ylabel("Value (%)")

    # ===== 在 CAGUI 子图上做红色高亮标注 =====
    if dataset == "CAGUI":
        # 1) GUI-Owl-7B: 连接 EM ↔ IDEAL, 标注 RG
        idx_owl7 = sub.index[sub["Model"] == "GUI-Owl-7B"]
        if len(idx_owl7) == 1:
            i = int(idx_owl7[0])
            xi = x[i]
            em_i = float(sub.loc[i, "EM"])
            ideal_i = float(sub.loc[i, "IDEAL"])
            ax.plot([xi, xi], [ideal_i, em_i], color="red", linewidth=2)
            mid_y = (ideal_i + em_i) / 2.0
            ax.text(xi + 0.06, mid_y-0.6, "RG", color="red", va="center")

        # 2) GUI-Owl-32B: 连接 GTA ↔ IDEAL, 标注 EG
        idx_owl32 = sub.index[sub["Model"] == "GUI-Owl-32B"]
        if len(idx_owl32) == 1:
            i = int(idx_owl32[0])
            xi = x[i]
            gta_i = float(sub.loc[i, "GTA"])
            ideal_i = float(sub.loc[i, "IDEAL"])
            ax.plot([xi, xi], [ideal_i, gta_i], color="red", linewidth=2)
            mid_y = (ideal_i + gta_i) / 2.0
            ax.text(xi-0.4, mid_y, "EG", color="red", va="center")

# ===== 共享图例（顺序：GTA、EM、IDEAL）=====
handles = [line_handles[k] for k in order_labels if k in line_handles]
fig.legend(handles, order_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

# plt.suptitle("Spline plots of GTA, EM, and IDEAL (EM−RG) with annotations", y=1.06)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("spline_em_gta_ideal_annotated.png", dpi=300, bbox_inches="tight")
plt.savefig("spline_em_gta_ideal_annotated.pdf", bbox_inches="tight")
plt.show()
