import json
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import pandas as pd

# Set publication-quality figure parameters
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": False,  # Set to True if LaTeX is available
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
    }
)

# Use clean style
sns.set_theme(style="whitegrid")

# Read scores (truncate decimals)
scores = []
with open(
    "/gs/bs/tga-NII-LLM/datasets/raw/pretrain/swallow-code-v0.3-merged/swallow-code-v0.3-no-repet.jsonl", "r"
) as f:
    for line in f:
        data = json.loads(line)
        if "score" in data:
            score = data["score"]
            scores.append(int(score))  # Truncate decimals

# Aggregate score distribution
score_counts = collections.Counter(scores)
bins = list(range(0, 11))  # Scores 0 to 10
frequencies = [score_counts.get(b, 0) for b in bins]

# Save to CSV
score_data = pd.DataFrame({"Score": bins, "Sample Count": frequencies})
score_data.to_csv("score_distribution.csv", index=False)

# Colors for bars
colors = ["gray" if b < 6 else "orange" for b in bins]
axis_color = "#000000"  # Black for axes

# Create figure
plt.figure(figsize=(6, 4), dpi=300, facecolor="white")
ax = plt.gca()

# Plot bar chart
plt.bar(bins, frequencies, color=colors, edgecolor="black", linewidth=1.5)

# Add threshold line
plt.axvline(5.5, color="red", linestyle="--", label="Score >= 6 threshold", linewidth=2.0, alpha=0.5)

# Grid styling
plt.grid(True, linestyle="--", alpha=0.2, color="#cccccc")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2.5)
ax.spines["bottom"].set_linewidth(2.5)
ax.spines["left"].set_color(axis_color)
ax.spines["bottom"].set_color(axis_color)
ax.tick_params(axis="both", color=axis_color, width=2.0)

# Labels and ticks
plt.xlabel("Score", fontsize=24, fontweight="bold", color="#000000")
plt.ylabel("Sample Count", fontsize=24, fontweight="bold", color="#000000")
plt.xticks(bins, [f"{b}" for b in bins], rotation=0, color="#000000", fontweight="bold")
plt.margins(x=0.01)

# Legend
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.3),
    ncol=1,
    frameon=True,
    fancybox=True,
    shadow=False,
    framealpha=0.9,
    edgecolor="#dddddd",
)

# Tight layout with increased padding
plt.tight_layout(pad=0.1)

# Save plot
plt.savefig("score_distribution.png", bbox_inches="tight", dpi=300)

# Show plot
plt.show()
