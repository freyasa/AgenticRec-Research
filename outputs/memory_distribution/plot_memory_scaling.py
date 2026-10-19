"""Plot NDCG@10 against the maximum memory length recorded in record.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "imgs" / "memory_scaling_ndcg10.png"

lengths = [50, 100, 150, 200, 300]
series = {
    "BaseRec Text": [0.4384, 0.3564, 0.3863, 0.3835, 0.3950],
    "BaseRec JSON": [0.3625, 0.4281, 0.3659, 0.4216, 0.3950],
    "AgentCF": [0.4383, 0.3947, 0.3370, 0.4059, 0.4229],
}

colors = ["#3B82F6", "#F59E0B", "#10B981"]
markers = ["o", "s", "^"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8FAFC")

for (name, values), color, marker in zip(series.items(), colors, markers):
    ax.plot(
        lengths,
        values,
        label=name,
        color=color,
        marker=marker,
        linewidth=2.2,
        markersize=7,
    )

ax.set_title("Memory Length Scaling", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Maximum memory length (words)")
ax.set_ylabel("NDCG@10")
ax.set_xticks(lengths)
ax.set_ylim(0.32, 0.46)
ax.grid(True, color="#DCE3EC", linewidth=0.8, alpha=0.9)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))

fig.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(OUTPUT)
