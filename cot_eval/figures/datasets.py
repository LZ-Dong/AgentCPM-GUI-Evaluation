import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Global font setting: Times New Roman
# -------------------------
# plt.rcParams["font.family"] = "Times New Roman"

# -------------------------
# Data
# -------------------------
datasets = {
    "AITZ": {
        "CLICK": 53.00,
        "STOP": 12.00,
        "SCROLL": 13.50,
        "INPUT": 12.00,
        "PRESS": 9.50,
        "LONG POINT": 0.00,
    },
    "CAGUI": {
        "CLICK": 65.50,
        "STOP": 14.00,
        "SCROLL": 4.00,
        "INPUT": 13.50,
        "PRESS": 0.00,
        "LONG POINT": 3.00,
    },
    "AndroidControl": {
        "CLICK": 53.00,
        "STOP": 18.00,
        "SCROLL": 14.50,
        "INPUT": 8.50,
        "PRESS": 6.00,
        "LONG POINT": 0.00,
    },
}
# datasets = {
#     "AITZ": {
#         "CLICK": 57.92,
#         "STOP": 10.67,
#         "SCROLL": 12.72,
#         "INPUT": 10.58,
#         "NO ACTION": 0.00,
#         "PRESS": 8.11,
#         "LONG POINT": 0.00,
#     },
#     "CAGUI": {
#         "CLICK": 71.68,
#         "STOP": 13.29,
#         "LONG POINT": 0.55,
#         "SCROLL": 1.75,
#         "INPUT": 12.71,
#         "NO ACTION": 0.00,
#         "PRESS": 0.00,
#         # "LONG POINT": 0.55,
#     },
#     "AndroidControl": {
#         "CLICK": 54.17,
#         "STOP": 16.53,
#         "SCROLL": 12.76,
#         "INPUT": 6.47,
#         "NO ACTION": 6.13,
#         "PRESS": 3.66,
#         "LONG POINT": 0.00,
#     },
# }
all_categories = ["CLICK", "STOP", "SCROLL", "INPUT", "NO ACTION", "PRESS", "LONG POINT"]

# Optional: weight each dataset's sector by its dataset size (e.g., number of samples).
# Set to a dict like {"AITZ": 1200, "CAGUI": 800, "AndroidControl": 500}.
# Leave as None to keep equal-sized sectors.
# dataset_sizes = {"AITZ": 4724, "CAGUI": 4516, "AndroidControl": 10161}
dataset_sizes = None

# -------------------------
# Styling
# -------------------------
tableau = {
    "blue":  "#4E79A7",
    "orange":"#F28E2B",
    "teal":  "#76B7B2",
    "green": "#59A14F",
    "red":   "#E15759",
    "purple":"#B07AA1",
    "brown": "#9C755F",
    "pink":  "#FF9DA7",
    "yellow":"#EDC948",
    "gray":  "#BAB0AC",
}

sector_colors = {
    "AITZ": tableau["blue"],
    "CAGUI": tableau["orange"],
    "AndroidControl": tableau["teal"],
}

bar_edgecolor = "#333333"

# -------------------------
# Layout parameters
# -------------------------
inner_radius = 2.1          # enlarged inner radius
bar_max_height = 1.8
category_bar_ratio = 0.8
label_radius = 1.1         # dataset label position inside sectors

global_max = max(v for d in datasets.values() for v in d.values())

# -------------------------
# Build polar chart
# -------------------------
fig = plt.figure(figsize=(9.2, 9.2))
ax = plt.subplot(111, polar=True)

dataset_names = list(datasets.keys())
num_datasets = len(dataset_names)

# Compute sector widths (equal or size-weighted)
if dataset_sizes:
    sizes = np.array([max(0.0, float(dataset_sizes.get(n, 0.0))) for n in dataset_names], dtype=float)
    if sizes.sum() > 0:
        sector_fracs = sizes / sizes.sum()
    else:
        sector_fracs = np.full(num_datasets, 1.0 / num_datasets)
else:
    sector_fracs = np.full(num_datasets, 1.0 / num_datasets)

sector_widths = 2 * np.pi * sector_fracs
sector_starts = np.concatenate(([0.0], np.cumsum(sector_widths)[:-1]))

# Orientation
ax.set_theta_offset(np.pi / 2.0)
ax.set_theta_direction(-1)

# Inner sectors
sector_handles = []
for i, ds_name in enumerate(dataset_names):
    theta_center = sector_starts[i] + sector_widths[i] / 2.0
    bars = ax.bar(
        [theta_center],
        [inner_radius],
        width=[sector_widths[i]],
        bottom=0.0,
        align="center",
        edgecolor=bar_edgecolor,
        linewidth=0.8,
        color=sector_colors[ds_name],
        alpha=0.95
    )
    sector_handles.append(bars[0])

# Outer bars and radially inward-pointing labels (skip zeros)
label_pad = 0.12  # radial padding for text above bar
for i, ds_name in enumerate(dataset_names):
    start = sector_starts[i]
    gap = 0.015 * sector_widths[i]
    usable_width = sector_widths[i] - 2 * gap

    nonzero_items = [(cat, val) for cat, val in datasets[ds_name].items() if val > 0]
    if not nonzero_items:
        continue

    n_nonzero = len(nonzero_items)
    cell_width = usable_width / n_nonzero
    bar_width = cell_width * category_bar_ratio

    for k, (cat, value) in enumerate(nonzero_items):
        theta_center = start + gap + (k + 0.5) * cell_width
        height = (value / global_max) * bar_max_height if global_max > 0 else 0.0

        # Bar (solid, sector color)
        ax.bar(
            [theta_center],
            [height],
            width=[bar_width],
            bottom=inner_radius,
            align="center",
            edgecolor=bar_edgecolor,
            linewidth=0.6,
            color=sector_colors[ds_name],
            alpha=0.98,
        )

        ax.text(
            theta_center,
            inner_radius + height + label_pad + 0.5,
            f"{cat}\n{value:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            clip_on=False
        )

# Dataset labels
for i, ds_name in enumerate(dataset_names):
    theta_center = sector_starts[i] + sector_widths[i] / 2.0
    ax.text(theta_center, label_radius, ds_name,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="#222222")

# Legend for datasets (colors)
# legend_datasets = ax.legend(
#     sector_handles,
#     dataset_names,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.06),
#     ncol=3,
#     frameon=False,
#     title="Datasets (inner sectors)"
# )

# Turn off all axes/ticks/grids
ax.set_axis_off()
ax.grid(False)

# Limits
ax.set_ylim(0, inner_radius + bar_max_height + 0.7)

# ax.set_title("Action-Space Distribution per Dataset",
#              pad=20, fontsize=14, fontweight="bold")

plt.show()
# save
fig.savefig("figures/dataset_action_space_distribution.png", dpi=300, bbox_inches="tight")