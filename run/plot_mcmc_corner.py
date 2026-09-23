"""Corner plot + split-half convergence check for the RWM chain."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import corner

import sys
NPZ = sys.argv[1] if len(sys.argv) > 1 else "/scratch/e1498138/eGB-multi/hmcnc_mcmc_min.npz"
OUT = "/home/svu/e1498138/localgit/eGB-multi/run/plots/" + __import__("os").path.basename(NPZ).replace(".npz", "_corner.png")
NAMES = ["eccentricity", r"$m_1\ [M_\odot]$", r"$m_2\ [M_\odot]$"]

d = np.load(NPZ)
chain = d["chain"]
truth = d["truth"]
burn = len(chain) // 5
post = chain[burn:]
print(f"{len(chain)} steps, discarding {burn} as burn-in -> {len(post)} samples")
print(f"unique states: {len(np.unique(post, axis=0))}/{len(post)}")

# --- split-half convergence check
a, b = np.array_split(post, 2)
print("\nsplit-half comparison:")
for i, n in enumerate(["e ", "m1", "m2"]):
    print(f"  {n}: 1st {a[:, i].mean():.5f} +/- {a[:, i].std():.5f} | "
          f"2nd {b[:, i].mean():.5f} +/- {b[:, i].std():.5f}")
print(f"  e 90% UL: 1st {np.percentile(a[:, 0], 90):.5f} | "
      f"2nd {np.percentile(b[:, 0], 90):.5f}")

# --- integrated autocorrelation (crude: first lag where ACF drops below 0.1)
print("\nautocorrelation length (ACF < 0.1):")
for i, n in enumerate(["e ", "m1", "m2"]):
    x = post[:, i] - post[:, i].mean()
    acf = np.correlate(x, x, mode="full")[len(x) - 1:]
    acf /= acf[0]
    tau = int(np.argmax(acf < 0.1)) if np.any(acf < 0.1) else len(acf)
    print(f"  {n}: ~{tau} steps  -> ~{len(post) // max(tau, 1)} effective samples")

# --- corner
fig = corner.corner(
    post,
    labels=NAMES,
    truths=truth,
    truth_color="#d1582a",
    color="#2f6fd0",
    show_titles=True,
    title_fmt=".4f",
    quantiles=[0.16, 0.5, 0.84],
    title_kwargs={"fontsize": 10},
    label_kwargs={"fontsize": 11},
    hist_kwargs={"linewidth": 1.5},
    plot_datapoints=False,
    fill_contours=True,
    levels=(0.68, 0.95),
    smooth=0.8,
)
fig.suptitle(
    f"HM Cnc RWM posterior - {len(post)} samples",
    fontsize=13, y=1.02)

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print("\nwrote", OUT)
