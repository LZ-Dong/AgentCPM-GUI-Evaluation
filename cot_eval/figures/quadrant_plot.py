# quadrant_plot_full_noborder.py
# -*- coding: utf-8 -*-
"""
完全无边框象限图
- 矩形无边框
- 图例无边框
- 去掉坐标轴四周的脊线
"""

from typing import Dict, Tuple, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_quadrant(
    counts: Dict[str, float],
    colors: Dict[str, str],
    labels: Dict[str, str],
    *,
    area_scale: float = 0.04,
    squares: bool = True,
    rect_ratio: float = 1.6,
    single_axes: bool = True,
    top_title: str = "GTA=1",
    bottom_title: str = "GTA=0",
    right_title: str = "EM=0",
    left_title: str = "EM=1",
    title_fontsize: int = 10,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_fontsize: int = 10,
    figsize: Tuple[float, float] = (6, 5),
    dpi: int = 160,
    save_path: Optional[str] = None,
) -> plt.Axes:
    def size_from_count(c: float):
        area = max(c, 0) * area_scale
        if squares:
            s = math.sqrt(area)
            return s, s
        else:
            h = math.sqrt(area / rect_ratio)
            w = rect_ratio * h
            return w, h

    size = {k: size_from_count(counts[k]) for k in ("Q1", "Q2", "Q3", "Q4")}
    max_extent = max(max(w, h) for (w, h) in size.values())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 中心线
    if single_axes:
        ax.axhline(0, color="black", lw=1)
        ax.axvline(0, color="black", lw=1)

    # Q2 (右上角，原Q1位置)
    w, h = size["Q2"]
    ax.add_patch(Rectangle((0, 0), w, h, facecolor=colors["Q2"], alpha=0.85,
                           edgecolor="none"))
    ax.text(0.5 * w, 0.5 * h, f"{counts['Q2']:.2f}",
            ha="center", va="center", fontsize=10, color="black")

    # Q1 (左上角，原Q2位置)
    w, h = size["Q1"]
    ax.add_patch(Rectangle((-w, 0), w, h, facecolor=colors["Q1"], alpha=0.85,
                           edgecolor="none"))
    ax.text(-0.5 * w, 0.5 * h, f"{counts['Q1']:.2f}",
            ha="center", va="center", fontsize=10, color="black")

    # Q4 (左下角，原Q3位置)
    w, h = size["Q4"]
    ax.add_patch(Rectangle((-w, -h), w, h, facecolor=colors["Q4"], alpha=0.85,
                           edgecolor="none"))
    ax.text(-0.5 * w, -0.5 * h, f"{counts['Q4']:.2f}",
            ha="center", va="center", fontsize=10, color="black")

    # Q3 (右下角，原Q4位置)
    w, h = size["Q3"]
    ax.add_patch(Rectangle((0, -h), w, h, facecolor=colors["Q3"], alpha=0.85,
                           edgecolor="none"))
    ax.text(0.5 * w, -0.5 * h, f"{counts['Q3']:.2f}",
            ha="center", va="center", fontsize=10, color="black")

    # 上下标签：上标签高于y轴顶部，下标签低于y轴底部
    ax.text(0, max_extent * 1.6, top_title, ha="center", va="bottom", fontsize=title_fontsize)
    ax.text(0, -max_extent * 1.6, bottom_title, ha="center", va="top", fontsize=title_fontsize)
    # 左右标签：略低于x轴
    ax.text(max_extent * 1.4, -0.2, right_title, ha="left", va="center", fontsize=title_fontsize)
    ax.text(-max_extent * 1.4, -0.2, left_title, ha="right", va="center", fontsize=title_fontsize)

    # 图例
    if show_legend:
        proxies = [Rectangle((0, 0), 1, 1, facecolor=colors[q], alpha=0.85,
                             edgecolor="none") for q in ("Q1", "Q2", "Q3", "Q4")]
        ax.legend(proxies, [labels[q] for q in ("Q1", "Q2", "Q3", "Q4")],
                  loc=legend_loc, fontsize=legend_fontsize, frameon=False)

    # 范围与刻度
    ax.set_xlim(-max_extent * 1.6, max_extent * 1.6)
    ax.set_ylim(-max_extent * 1.6, max_extent * 1.6)
    ax.set_xticks([])
    ax.set_yticks([])

    # 去掉整张图的四周边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return ax


# 示例
if __name__ == "__main__":
    counts = {"Q1": 65.76, "Q2": 8.69, "Q3": 22.14, "Q4": 3.41}
    colors = {"Q1": "#4A90E2", "Q2": "#6FA8DC", "Q3": "#AED6F1", "Q4": "#A9CCE3"}
    labels = {
        "Q1": "Correct CoT & Action",
        "Q2": "Correct CoT, Wrong Action",
        "Q3": "Wrong CoT & Action",
        "Q4": "Wrong CoT, Correct Action",
    }

    draw_quadrant(counts, colors, labels,
                  area_scale=0.04,
                  save_path="quadrant_plot.png")

    # plt.show()
