import numpy as np
import matplotlib.pyplot as plt

labels = ["AgentCPM-GUI-8B", "UI-TARS-1.5-7B", "GUI-Owl-7B"]
series = {
    "AndroidControl":    [88.89, 92.90, 93.83],
    "CAGUI": [82.84, 89.88, 78.89],
    "AITZ":  [77.36, 86.93, 84.00],
}

# 角度（三个维度 + 闭合点）
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 创建画布：1行3列子图
fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(9,3), dpi=300)

for ax, (name, vals) in zip(axes, series.items()):
    vals = vals + vals[:1]  # 闭合

    # 设置角度和刻度
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2*np.pi, N, endpoint=False))
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis='x', pad=18)  # 调整标签与图的距离
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_rlabel_position(0)

    # 绘制
    ax.plot(angles, vals, linewidth=1.5, label=name)
    ax.fill(angles, vals, alpha=0.25)
    # ax.set_title(name, fontsize=10, pad=12)
    ax.text(0.5, -0.2, name, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="black")

plt.tight_layout()
plt.savefig("radar_subplots.pdf")
plt.savefig("radar_subplots.png", dpi=300)
plt.show()
