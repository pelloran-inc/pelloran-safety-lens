#!/usr/bin/env python3
# Copyright (C) 2026 Pelloran Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Render the README hero image: trap-fall rate HN@10, raw cosine vs the lens,
four domains (5k pools). The lens is 0.0000 everywhere; raw cosine leaks for
81-100% of queries. Values from results/raw_cosine/ and results/pool_sizes/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # package root
domains = ["Legal", "Medical", "Patent", "Financial"]
raw = [0.8127, 0.9967, 0.9967, 0.9900]   # raw cosine HN@10 at 5k
lens = [0.0, 0.0, 0.0, 0.0]              # the safety lens HN@10

y = np.arange(len(domains))[::-1]        # Legal on top
fig, ax = plt.subplots(figsize=(7.6, 3.1), constrained_layout=True)

ax.barh(y, raw, 0.62, color="#D1495B", edgecolor="black", linewidth=0.4, label="without the lens (raw cosine)")
for yi, v in zip(y, raw):
    ax.annotate(f"{v*100:.0f}%", xy=(v, yi), xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color="#D1495B", fontweight="bold")

# the lens: a marker at 0; its "0.0000 everywhere" message lives in the legend
ax.scatter([0]*len(y), y, s=48, color="#4C78A8", zorder=5, edgecolor="black", linewidth=0.5,
           label="with the safety lens  —  0.0000 in all four domains")

ax.set_yticks(y, domains, fontsize=11)
ax.set_xlim(0, 1.12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of queries with the meaning-reversing trap in the top-10  (HN@10, 5k pool)", fontsize=10)
ax.grid(axis="x", linestyle=":", alpha=0.4); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, frameon=False, fontsize=9.5)

out = HERE / "assets" / "hero_hn.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200)
print(f"wrote {out}")
