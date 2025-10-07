import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.size": 16,        # 控制所有文字默认大小
    # "axes.titlesize": 22,   # 子图标题
    # "axes.labelsize": 22,   # 坐标轴标签
    # "xtick.labelsize": 18,  # x轴刻度
    # "ytick.labelsize": 18,  # y轴刻度
    "legend.fontsize": 10,  # 图例字体
    # "legend.title_fontsize": 20  # 图例标题
})

# ===== 1) 数据：来自你的 Table 1 =====
rows = [
    # (Dataset, Model, EG, RG)
    ("AITZ", "AgentCPM-GUI-8B", 3.51, 9.38),
    ("AITZ", "UI-TARS-7B-SFT", 7.73, 9.44),
    ("AITZ", "UI-TARS-7B-DPO", 8.48, 10.57),
    ("AITZ", "UI-TARS-1.5-7B", 11.85, 4.69),
    ("AITZ", "GUI-Owl-7B", 10.29, 5.12),
    ("AITZ", "GUI-Owl-32B", 12.19, 5.76),
    ("CAGUI", "AgentCPM-GUI-8B", 2.28, 8.13),
    ("CAGUI", "UI-TARS-7B-SFT", 9.08, 2.57),
    ("CAGUI", "UI-TARS-7B-DPO", 10.35, 2.44),
    ("CAGUI", "UI-TARS-1.5-7B", 15.32, 2.06),
    ("CAGUI", "GUI-Owl-7B", 21.75, 2.88),
    ("CAGUI", "GUI-Owl-32B", 18.29, 2.64),
    ("AndroidControl", "AgentCPM-GUI-8B", 8.68, 3.41),
    ("AndroidControl", "UI-TARS-7B-SFT", 9.18, 2.45),
    ("AndroidControl", "UI-TARS-7B-DPO", 9.15, 2.85),
    ("AndroidControl", "UI-TARS-1.5-7B", 5.53, 5.41),
    ("AndroidControl", "GUI-Owl-7B", 12.31, 4.74),
    ("AndroidControl", "GUI-Owl-32B", 10.43, 6.18),
]
df = pd.DataFrame(rows, columns=["Dataset", "Model", "EG", "RG"])
df["Delta"] = df["EG"] - df["RG"]

# ===== 2) 可视化映射 =====
# 颜色：按数据集
colors = {"AITZ": "orange", "CAGUI": "blue", "AndroidControl": "green"}
# 点形：按模型
markers = {
    "AgentCPM-GUI-8B": "o",
    "UI-TARS-7B-SFT": "s",
    "UI-TARS-7B-DPO": "^",
    "UI-TARS-1.5-7B": "D",
    "GUI-Owl-7B": "P",
    "GUI-Owl-32B": "X",
}

# ===== 3) 绘制三个并列子图 =====
datasets = ["AITZ", "CAGUI", "AndroidControl"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# 计算全局轴限，保持三个子图尺度一致
global_max = max(df["EG"].max(), df["RG"].max())
pad = 2
axis_max = np.ceil(global_max + pad)

for ax, dataset in zip(axes, datasets):
    subdf = df[df["Dataset"] == dataset]
    # 绘制点
    for _, row in subdf.iterrows():
        ax.scatter(
            row["RG"], row["EG"],
            c=colors[dataset],
            marker=markers[row["Model"]],
            s=100, edgecolor="k", alpha=0.85,
            label=row["Model"]  # 用于后面去重图例
        )
    # y=x 参考线
    ax.plot([0, axis_max], [0, axis_max], "--", color="gray", linewidth=1)
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_title(dataset, fontsize=13)
    ax.set_xlabel("RG (%)")
    ax.set_ylabel("EG (%)")

# ===== 4) 双图例：颜色=数据集；点形=模型 =====
# 左图放“数据集”颜色图例
handles_ds = [plt.Line2D([0],[0], marker='o', color='w',
                         markerfacecolor=c, markeredgecolor='k', markersize=10,
                         linestyle='None', label=ds)
              for ds, c in colors.items()]
legend_ds = axes[0].legend(handles=handles_ds, title="Dataset", loc="upper left")
axes[0].add_artist(legend_ds)

# 右图放“模型”形状图例
handles_models = [plt.Line2D([0],[0], marker=m, color='k', linestyle='None',
                             markersize=10, label=model)
                  for model, m in markers.items()]
axes[2].legend(handles=handles_models, title="Model", loc="lower right", ncol=2)

# plt.suptitle("EG vs RG across datasets and models", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ===== 5) 保存 =====
plt.savefig("eg_vs_rg_subplots.png", dpi=300)
plt.savefig("eg_vs_rg_subplots.pdf")
plt.show()

# ===== 6) 可选：打印统计，便于写图注 =====
N = len(df)
num_pos = int((df["Delta"] > 0).sum())
median_delta = df["Delta"].median()
print(f"EG > RG: {num_pos}/{N} = {num_pos/N:.1%}; median Δ(EG-RG) = {median_delta:.2f}")
