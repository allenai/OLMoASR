# %%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# # Short-form Speech Recognition
# %%
y_lists = [
    [20.14, 16.87, 13.71, 12.44, 12.19],
    [20.51, 16.59, 13.80, 12.77, 13.00],
    [37.15, 20.90, 17.00, 17.43, 14.53],
    [23.24, 18.25, 16.12],
]

x_lists = [
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M"],
]

labels = [
    "Whisper",
    "OLMoASR trained on OLMoASR-Mix",
    "OLMoASR trained on OLMoASR-Pool",
    "OLMoASR trained on YODAS",
]
# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 7))

# Optional: color by group
colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]
line_styles = ["-", "--", "-.", ":"]
custom_lines = []
for i in range(4):
    ax.scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    ax.plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=2
    )

    line = Line2D(
        [0],
        [0],
        color=colors[i],
        linestyle=line_styles[i],
        label=labels[i],
        linewidth=2,
    )
    custom_lines.append(line)

ax.set_xlabel("Model Size", fontsize=16)
ax.set_ylabel("Average WER", fontsize=16)
ax.set_title("Short-form Speech Recognition", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14, shadow=True, fancybox=True, handles=custom_lines)
ax.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig("logs/plots/short_wer.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # Long-form Speech Recognition
# %%
y_lists = [
    [16.64, 13.24, 11.24, 10.53, 10.43],
    [15.63, 12.87, 11.46, 11.00, 11.37],
    [23.46, 17.99, 15.79, 16.45, 13.33],
    [16.93, 15.22, 13.41],
]

x_lists = [
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M"],
]

labels = [
    "Whisper",
    "OLMoASR trained on OLMoASR-Mix",
    "OLMoASR trained on OLMoASR-Pool",
    "OLMoASR trained on YODAS",
]
# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 7))

# Optional: color by group
colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]
line_styles = ["-", "--", "-.", ":"]
custom_lines = []
for i in range(4):
    ax.scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    ax.plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=2
    )

    line = Line2D(
        [0],
        [0],
        color=colors[i],
        linestyle=line_styles[i],
        label=labels[i],
        linewidth=2,
    )
    custom_lines.append(line)

ax.set_xlabel("Model Size", fontsize=16)
ax.set_ylabel("Average WER", fontsize=16)
ax.set_title("Long-form Speech Recognition", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14, shadow=True, fancybox=True, handles=custom_lines)
ax.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig("logs/plots/long_wer.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # Data Scaling
# %%
y_lists = [
    [17.29, 16.41, 16.59, 16.49, 16.72, 15.18],
    [13.48, 12.84, 12.87, 12.81, 12.91, 12.89],
]
x_lists = [
    [0.048, 0.2114578549, 0.4211429212, 0.6537791422, 0.8450538568, 1.0],
    [0.048, 0.2114578549, 0.4211429212, 0.6537791422, 0.8450538568, 1.0],
]

labels = ["Short-form", "Long-form"]

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 7))

# Optional: color by group
colors = ["#E66100", "#5D3A9B"]
line_styles = ["-.", ":"]
custom_lines = []
for i in range(2):
    ax.scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    ax.plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=2
    )

    line = Line2D(
        [0],
        [0],
        color=colors[i],
        linestyle=line_styles[i],
        label=labels[i],
        linewidth=2,
    )
    custom_lines.append(line)

ax.set_xlabel("Fraction of OLMoASR-Mix used for training", fontsize=16)
ax.set_ylabel("Average WER", fontsize=16)
ax.set_title("English Speech Recognition", fontsize=16)
ax.set_xticks([0.25, 0.5, 0.75, 1.0])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14, shadow=True, fancybox=True, handles=custom_lines)
ax.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig("logs/plots/data_scale.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # OWSM vs OLMoASR (short-form)
# %%
y_lists = [[19.37, 16.58, 14.84], [18.11, 15.99, 14.97]]
x_lists = [[113, 226, 452], [113, 226, 452]]

labels = ["Trained on OWSM-Eng", "Trained on OLMoASR-Mix"]

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 7))

colors = ["#1A85FF", "#D41159"]
line_styles = ["-.", ":"]
custom_lines = []
for i in range(2):
    ax.scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    ax.plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=2
    )

    line = Line2D(
        [0],
        [0],
        color=colors[i],
        linestyle=line_styles[i],
        label=labels[i],
        linewidth=2,
    )
    custom_lines.append(line)

ax.set_xlabel("Total data seen (in K hours)", fontsize=16)
ax.set_ylabel("Average WER", fontsize=16)
ax.set_title("Short-form Speech Recognition", fontsize=16)
ax.set_xticks([100, 200, 300, 400, 500])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14, shadow=True, fancybox=True, handles=custom_lines)
ax.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig("logs/plots/owsm_olmoasr.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # OWSM vs OLMoASR (long-form)
# %%
y_lists = [[19.31, 17.49, 16.14], [13.76, 12.25, 11.91]]
x_lists = [[113, 226, 452], [113, 226, 452]]

labels = ["Trained on OWSM-Eng", "Trained on OLMoASR-Mix"]

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 7))

colors = ["#1A85FF", "#D41159"]
line_styles = ["-.", ":"]
custom_lines = []
for i in range(2):
    ax.scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    ax.plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=2
    )

    line = Line2D(
        [0],
        [0],
        color=colors[i],
        linestyle=line_styles[i],
        label=labels[i],
        linewidth=2,
    )
    custom_lines.append(line)

ax.set_xlabel("Total data seen (in K hours)", fontsize=16)
ax.set_ylabel("Average WER", fontsize=16)
ax.set_title("Long-form Speech Recognition", fontsize=16)
ax.set_xticks([100, 200, 300, 400, 500])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14, shadow=True, fancybox=True, handles=custom_lines)
ax.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig("logs/plots/owsm_olmoasr_long.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # OWSM vs OLMoASR (combined short-form and long-form)
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Short-form WER
y_lists_short = [[19.37, 16.58, 14.84], [18.11, 15.99, 14.97]]
x_lists = [[113, 226, 452], [113, 226, 452]]

# Long-form WER
y_lists_long = [[19.31, 17.49, 16.14], [13.76, 12.25, 11.91]]

labels = ["Trained on OWSM-Eng", "Trained on OLMoASR-Mix"]
colors = ["#1A85FF", "#D41159"]
line_styles = ["-.", ":"]
custom_lines = []

# Short-form subplot
for i in range(2):
    axes[0].scatter(
        x_lists[i], y_lists_short[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    axes[0].plot(
        x_lists[i], y_lists_short[i], color=colors[i], linestyle=line_styles[i], linewidth=2
    )
    line = Line2D(
        [0], [0], color=colors[i], linestyle=line_styles[i], label=labels[i], linewidth=2
    )
    custom_lines.append(line)

axes[0].set_xlabel("Total data seen (in K hours)", fontsize=16)
axes[0].set_ylabel("Average WER", fontsize=16)
axes[0].set_title("Short-form Speech Recognition", fontsize=16)
axes[0].set_xticks([100, 200, 300, 400, 500])
axes[0].tick_params(labelsize=14)
axes[0].legend(fontsize=14, shadow=True, fancybox=True, handles=custom_lines)
axes[0].grid(True, linestyle="-", alpha=0.6)

# Long-form subplot
for i in range(2):
    axes[1].scatter(
        x_lists[i], y_lists_long[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    axes[1].plot(
        x_lists[i], y_lists_long[i], color=colors[i], linestyle=line_styles[i], linewidth=2
    )

axes[1].set_xlabel("Total data seen (in K hours)", fontsize=16)
axes[1].set_ylabel("Average WER", fontsize=16)
axes[1].set_title("Long-form Speech Recognition", fontsize=16)
axes[1].set_xticks([100, 200, 300, 400, 500])
axes[1].tick_params(labelsize=14)
axes[1].legend(fontsize=14, shadow=True, fancybox=True, handles=custom_lines)
axes[1].grid(True, linestyle="-", alpha=0.6)

plt.tight_layout()
plt.savefig("logs/plots/owsm_olmoasr_combined.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # Short-form and Long-form Speech Recognition (Combined)
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Short-form Speech Recognition
y_lists = [
    [20.14, 16.87, 13.71, 12.44, 12.19],
    [20.51, 16.59, 13.80, 12.77, 13.00],
    [37.15, 20.90, 17.00, 17.43, 14.53],
    [23.24, 18.25, 16.12],
]
x_lists = [
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M"],
]
labels = [
    "Whisper",
    "OLMoASR + OLMoASR-Mix",
    "OLMoASR + no quality filtering subset",
    "OLMoASR + YODAS",
]
colors = ["#648FFF", "#F92E91", "#FF88CB", "#44843C"]
line_styles = ["-", "--", "-.", ":"]
custom_lines = []

for i in range(4):
    axes[0].scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    axes[0].plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=3
    )
    line = Line2D(
        [0],
        [0],
        color=colors[i],
        linestyle=line_styles[i],
        label=labels[i],
        linewidth=3,
    )
    custom_lines.append(line)

axes[0].set_xlabel("Model Size", fontsize=18)
axes[0].set_ylabel("Average WER", fontsize=18)
axes[0].set_title("Short-form Speech Recognition", fontsize=20)
axes[0].tick_params(labelsize=16)
axes[0].legend(fontsize=16, shadow=True, fancybox=True, handles=custom_lines)
axes[0].grid(True, linestyle="-", alpha=0.6)

# Plot 2: Long-form Speech Recognition
y_lists = [
    [16.64, 13.24, 11.24, 10.53, 10.43],
    [15.63, 12.87, 11.46, 11.00, 11.37],
    [23.46, 17.99, 15.79, 16.45, 13.33],
    [16.93, 15.22, 13.41],
]
x_lists = [
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M"],
]

for i in range(4):
    axes[1].scatter(
            x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
        )
    axes[1].plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=3
    )

axes[1].set_xlabel("Model Size", fontsize=18)
axes[1].set_ylabel("Average WER", fontsize=18)
axes[1].set_title("Long-form Speech Recognition", fontsize=20)
axes[1].tick_params(labelsize=16)
axes[1].legend(fontsize=16, shadow=True, fancybox=True, handles=custom_lines)
axes[1].grid(True, linestyle="-", alpha=0.6)

plt.tight_layout()
plt.savefig("logs/plots/short_long_wer_combined.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # Short-form Speech Recognition (LibriSpeech and average WER)
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Short-form Speech Recognition (LibriSpeech)
y_lists = [
    [5.6, 4.2, 3.1, 3.1, 2.7],
    [5.1, 3.7, 3.0, 3.5, 2.6],
    [6.0, 4.4, 3.3, 3.0, 3.0],
    [6.6, 5.2, 4.1],
]
x_lists = [
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M"],
]
labels = [
    "Whisper",
    "OLMoASR + OLMoASR-Mix",
    "OLMoASR + no quality filtering subset",
    "OLMoASR + YODAS",
]
colors = ["#648FFF", "#F92E91", "#FF88CB", "#44843C"]
line_styles = ["-", "--", "-.", ":"]
custom_lines = []

for i in range(4):
    axes[0].scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    axes[0].plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=3
    )
    line = Line2D(
        [0],
        [0],
        color=colors[i],
        linestyle=line_styles[i],
        label=labels[i],
        linewidth=3,
    )
    custom_lines.append(line)

axes[0].set_xlabel("Model Size", fontsize=18)
axes[0].set_ylabel("WER on LibriSpeech.test-clean", fontsize=18)
axes[0].set_title("Short-form Speech Recognition", fontsize=20)
axes[0].tick_params(labelsize=16)
axes[0].legend(fontsize=16, shadow=True, fancybox=True, handles=custom_lines)
axes[0].grid(True, linestyle="-", alpha=0.6)

# Plot 2: Short-form Speech Recognition
y_lists = [
    [20.14, 16.87, 13.71, 12.44, 12.19],
    [20.51, 16.59, 13.80, 12.77, 13.00],
    [37.15, 20.90, 17.00, 17.43, 14.53],
    [23.24, 18.25, 16.12],
]
x_lists = [
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M", "769M", "1550M"],
    ["39M", "74M", "244M"],
]

for i in range(4):
    axes[1].scatter(
        x_lists[i], y_lists[i], label=labels[i], s=100, alpha=0.9, color=colors[i]
    )
    axes[1].plot(
        x_lists[i], y_lists[i], color=colors[i], linestyle=line_styles[i], linewidth=3
    )

axes[1].set_xlabel("Model Size", fontsize=18)
axes[1].set_ylabel("Average WER", fontsize=18)
axes[1].set_title("Short-form Speech Recognition", fontsize=20)
axes[1].tick_params(labelsize=16)
axes[1].legend(fontsize=16, shadow=True, fancybox=True, handles=custom_lines)
axes[1].grid(True, linestyle="-", alpha=0.6)

plt.tight_layout()
plt.savefig("logs/plots/short_libri_wer_combined.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %% [markdown]
# # Effective and Relative Robustness
# %%
# Support plotting multiple regression lines
x_lists = [
    [5.10, 3.70, 3.04, 3.54, 2.62, 2.66],
    [5.60, 4.20, 3.10, 3.10, 2.70],
    # [4.6, 3.8, 3.4],
    # [2.9, 2.3, 2.0],
    [1.8, 6.0, 3.3, 2.7, 2.0, 1.9, 5.7, 3.3, 3.6, 3.0, 2.1],
]
y_lists = [
    [35.12, 29.22, 24.40, 22.73, 23.13, 22.73],
    [33.10, 29.04, 23.28, 21.40, 21.56],
    # [31.0, 27.7, 26.5],
    # [34.9, 31.1, 28.6],
    [38.64, 60.72, 51.38, 47.18, 41.12, 40.40, 58.06, 60.08, 63.46, 70.30, 54.98],
]
labels = [
    "Zero-shot OLMoASR models",
    "Zero-shot Whisper models",
    # "Zero-shot OLMoASR-244M (1-epoch)",
    # "Zero-shot OLMoASR-244M (on OWSM-Eng data)",
    "Supervised LibriSpeech models"
]
# colors = ["#1A85FF", "#D41159", "#FFB300", "#00A08B", "#B342FF"]
colors = ["#1A85FF", "#D41159", "#B342FF"]
# markers = ["o", "s", "D", "^", "v"]
markers = ["o", "s", "v"]

plt.figure(figsize=(10, 8))
for x, y, label, color in zip(x_lists, y_lists, labels, colors):
    sns.regplot(x=x, y=y, ci=97, label=label, scatter_kws={"s": 100}, color=color, marker=markers[labels.index(label)], line_kws={"linewidth": 3})

plt.xlabel("WER on LibriSpeech test-clean (%)", fontsize=18)
plt.ylabel("Average WER on AMI, CommonVoice, CHiME-6, CORAAL (%)", fontsize=18)
plt.title("Effective and Relative Robustness", fontsize=20)
plt.legend(fontsize=16, shadow=True, fancybox=True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig(
    "logs/plots/robustness.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()

# %%
# Support plotting multiple regression lines
x_lists = [
    [4.6, 3.8, 3.4],
    [2.9, 2.3, 2.0],
]
y_lists = [
    [32.1, 29.2, 27.3],
    [39.5, 35.3, 32.8],
]
labels = [
    "Zero-shot OLMoASR-244M (1-epoch)",
    "Zero-shot OLMoASR-244M (on OWSM-Eng data)",
]
colors = ["#FFB300", "#00A08B"]
markers = ["D", "^"]

plt.figure(figsize=(10, 8))
for x, y, label, color in zip(x_lists, y_lists, labels, colors):
    sns.regplot(x=x, y=y, ci=97, label=label, scatter_kws={"s": 100}, color=color, marker=markers[labels.index(label)], line_kws={"linewidth": 3})

plt.xlabel("WER on LibriSpeech test-clean (%)", fontsize=18)
plt.ylabel("Average WER on CHiME-6, CORAAL (%)", fontsize=18)
plt.title("Effective and Relative Robustness", fontsize=20)
plt.legend(fontsize=16, shadow=True, fancybox=True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig(
    "logs/plots/owsm_olmoasr_robustness.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
# %% [markdown]
# Quality-Diversity Trade Off
# %%
x_list = [
    53.68,
    42.19,
    58.04,
    58.70,
    51.48,
    54.67,
    51.37,
    39.94,
    50.79,
    68.02,
    60.07,
    56.69,
    46.96,
    43.95,
    48.16,
    46.65,
    27.94,
    41.34,
    43.65,
    45.21,
    44.27,
]
y_list = [
    27.27,
    26.80,
    30.86,
    30.43,
    27.38,
    31.19,
    29.57,
    28.31,
    29.14,
    33.96,
    23.41,
    22.96,
    23.58,
    22.35,
    22.81,
    22.17,
    21.55,
    23.08,
    21.78,
    21.70,
    21.34,
]
labels = [
    "Other manual-machine text comparison score",
    "Retain only mixed-case",
    "Remove repeating lines",
    "Removing upper-case, repeating lines",
    "Removing upper, lower-case, repeating lines",
    "seg_edit_dist <= 0.2, edit_dist <= 0.2",
    "seg_edit_dist <= 0.4, edit_dist <= 0.4",
    "seg_edit_dist <= 0.6, edit_dist <= 0.4",
    "seg_edit_dist <= 0.7, edit_dist <= 0.5",
    "Final filters",
]
# plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(figsize=(12, 10))
colors = [
    "#1A85FF",
    "#D41159",
    "#FFB300",
    "#00A08B",
    "#648FFF",
    "#785EF0",
    "#DC2626",
    "#FE6100",
    "#000DFF",
    "#00C4E6",
]

# Unique marker styles
markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "H"]

used_labels = set()

for i, (x, y) in enumerate(zip(x_list, y_list)):
    if i < 9:
        label = labels[0]
        color = colors[0]
        marker = markers[0]
    elif i == 9:
        label = labels[1]
        color = colors[1]
        marker = markers[1]
    elif i == 10:
        label = labels[2]
        color = colors[2]
        marker = markers[2]
    elif i == 11:
        label = labels[3]
        color = colors[3]
        marker = markers[3]
    elif i == 12:
        label = labels[4]
        color = colors[4]
        marker = markers[4]
    elif i > 12 and i < 16:
        label = labels[0]
        color = colors[0]
        marker = markers[0]
    elif i == 16:
        label = labels[5]
        color = colors[5]
        marker = markers[5]
    elif i == 17:
        label = labels[6]
        color = colors[6]
        marker = markers[6]
    elif i == 18:
        label = labels[7]
        color = colors[7]
        marker = markers[7]
    elif i == 19:
        label = labels[8]
        color = colors[8]
        marker = markers[8]
    elif i == 20:
        label = labels[9]
        color = colors[9]
        marker = markers[9]

    if label not in used_labels:
        ax.scatter(x, y, label=label, s=200, alpha=0.9, color=color, marker=marker)
        used_labels.add(label)
    else:
        ax.scatter(x, y, s=200, alpha=0.9, color=color)
    # ax.scatter(x, y, label=label, s=100, alpha=0.9, color=color)
    # plt.scatter(x, y, s=100, alpha=0.9, label=label)

ax.set_xlabel("Percent of Data Remaining (%)", fontsize=20)
ax.set_ylabel("Average WER across 10 datasets (%)", fontsize=20)
ax.set_title("Quality-Diversity Trade Off", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.legend(fontsize=16, shadow=True, fancybox=True)
ax.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig(
    "logs/plots/qual_div.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
# %%
