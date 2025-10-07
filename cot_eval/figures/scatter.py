import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# 需要手动微调的点（按需填写）positive metrics
# LABEL_OVERRIDES = {
#     "UI-TARS-72B-DPO": dict(dx=-60, dy=10, ha="center",  va="bottom"),
#     "UI-TARS-72B-SFT": dict(dx=-45, dy=25, ha="center",  va="bottom"),
#     "UI-TARS-7B-SFT": dict(dx=50, dy=-25, ha="center",  va="bottom"),
#     "UI-TARS-7B-DPO": dict(dx=-35, dy=10, ha="center",  va="bottom"),
# }
LABEL_OVERRIDES = {
    "UI-TARS-72B-SFT": dict(dx=-55, dy=10, ha="center",  va="bottom"),
    "UI-TARS-72B-DPO": dict(dx=65, dy=-60, ha="center",  va="bottom"),
    "UI-TARS-7B-DPO": dict(dx=-35, dy=10, ha="center",  va="bottom"),
    # "UI-TARS-7B-SFT": dict(dx=50, dy=-25, ha="center",  va="bottom"),
    # "UI-TARS-7B-DPO": dict(dx=-35, dy=10, ha="center",  va="bottom"),
}

# ---- Data ----
data = [
    ("UI-TARS-2B-SFT", 69.69, 79.39, 13.13, 3.42, 2, "SFT"),
    ("UI-TARS-7B-SFT", 72.18, 81.77, 12.81, 3.23, 7, "SFT"),
    ("UI-TARS-72B-SFT", 73.14, 83.09, 12.17, 2.22, 72, "SFT"),
    ("UI-TARS-7B-DPO", 66.91, 79.56, 15.40, 2.75, 7, "DPO"),
    ("UI-TARS-72B-DPO", 71.52, 82.86, 13.47, 2.13, 72, "DPO"),
]
df = pd.DataFrame(data, columns=["Model", "EM", "GTA", "EG", "RG", "ParamsB", "Type"])

# Bubble size scaling
valid_params = df["ParamsB"].dropna()
# sizes = (np.log10(df["ParamsB"].fillna(valid_params.min())) - np.log10(valid_params.min()))
# sizes = 1000 * (sizes / (sizes.max() if sizes.max() > 0 else 1) + 0.6)
# Bubble size scaling (linear)
valid_params = df["ParamsB"].dropna()
sizes = df["ParamsB"].fillna(valid_params.min())
# 归一化并放大到合适范围
sizes = 4000 * (sizes / sizes.max())
print(sizes)

# Colors
colors = {"SFT": "steelblue", "DPO": "darkorange"}

def plot_scatter(xcol, ycol, xlabel, ylabel, title, outfile):
    plt.figure(figsize=(7, 6))
    texts = []
    for t, sub in df.dropna(subset=[xcol, ycol]).groupby("Type"):
        plt.scatter(
            sub[xcol], sub[ycol],
            s=sizes[sub.index],
            c=colors.get(t, "gray"),
            marker="o",
            alpha=0.8,
            label=t
        )
        # 初始放左上（ha=right, va=bottom）
        # for i, r in sub.iterrows():
        #     label = f"{r['Model']}\n{xcol}={r[xcol]:.2f}\n{ycol}={r[ycol]:.2f}"
        #     texts.append(
        #         plt.text(r[xcol], r[ycol], label,
        #                  fontsize=9, ha="right", va="bottom")
        #     )
        texts_all, texts_auto = [], []   # texts_auto = 只对它们做 adjust_text（可选）
        ax = plt.gca()
        for i, r in sub.iterrows():
            label = f"{r['Model']}\n{xcol}={r[xcol]:.2f}\n{ycol}={r[ycol]:.2f}"
            ov = LABEL_OVERRIDES.get(r["Model"])
            if ov:  # —— 手动覆盖：像素偏移 + 自定义对齐 —— #
                txt = ax.annotate(
                    label, xy=(r[xcol], r[ycol]),
                    xytext=(ov.get("dx", 0), ov.get("dy", 0)),
                    textcoords="offset points",
                    ha=ov.get("ha", "right"),
                    va=ov.get("va", "bottom"),
                    fontsize=10
                )
                texts_all.append(txt)
                # 注意：不把它放进 texts_auto，避免被 adjust_text 再改动
            else:   # —— 保持原状：左上（ha=right, va=bottom），无偏移 —— #
                txt = plt.text(r[xcol], r[ycol], label, fontsize=10, ha="right", va="bottom")
                texts_all.append(txt)
                texts_auto.append(txt)  # 仅这些让 adjust_text 微调（可选）

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    fig = plt.gcf()
    fig_w, fig_h = fig.get_size_inches()
    ax = plt.gca()

    xmin, xmax = df[xcol].min() - 2, df[xcol].max() + 2
    ymin, ymax = df[ycol].min() - 2, df[ycol].max() + 2

    Wx = xmax - xmin
    Wy = ymax - ymin
    current_ratio = (Wx / Wy) * (fig_w / fig_h)
    target_ratio = 3 / 5

    if current_ratio < target_ratio:
        # 扩展 x 轴范围
        needed_Wx = target_ratio * Wy * (fig_h / fig_w)
        extra = (needed_Wx - Wx) / 2
        xmin -= extra
        xmax += extra

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ticks step=2
    plt.xticks(np.arange(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, 2))
    plt.yticks(np.arange(int(ax.get_ylim()[0]), int(ax.get_ylim()[1]) + 1, 2))


    # Legend bottom right
    plt.legend(title="", loc="lower right", scatterpoints=1, markerscale=0.25, fontsize=10)

    # 自动调整，尽量保持左上
    adjust_text(
        # texts,
        texts_auto, 
        autoalign='xy'  # 限制调整方向
    )

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()

# ----- 绘制 -----
# plot_scatter("EM", "GTA", "EM (%)", "GTA (%)",
#              "AndroidControl — Positive Metrics (Higher is Better)",
#              "scatter_positive_adjust.png")

plot_scatter("EG", "RG", "EG (%)", "RG (%)",
             "AndroidControl — Negative Metrics (Lower is Better)",
             "scatter_negative_adjust.png")