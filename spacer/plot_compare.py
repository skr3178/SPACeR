"""Overlay SPACeR training curves from multiple runs (β-sweep / ablation).

    python3 spacer/plot_compare.py <out.png> <label1>:<log1> ...    (host)

Each panel shows one line per run, coloured by run, with a legend. Same
6-metric layout as plot_curves.py (KL, log-LL, entropy, total loss, r_task,
value loss, |g|). Default output → repo-root `plots/`.

Example
-------
  python3 spacer/plot_compare.py plots/bsweep_compare.png \\
      "β=0.01:spacer/logs/run_bsweep_b0.01_20260522_133736.log" \\
      "β=0.10:spacer/logs/run_bsweep_b0.1_20260522_185949.log" \\
      "β=1.00:spacer/logs/run_bsweep_b1.0_20260522_185949.log"
"""
import sys, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pat = re.compile(
    r"\[(?:ABL )?β=([\d.]+) W=(\d+)\] it(\d+): "
    r"r_task=([-+\d.]+) r_h=([-+\d.]+) KL=([-+\d.]+) "
    r"H=([-+\d.]+) pg=([-+\d.]+) vL=([-+\d.]+) loss=([-+\d.]+) \|g\|=([-+\d.]+)")

KEYS = ["r_task", "r_h", "KL", "H", "pg", "vL", "loss", "g"]


def parse_log(path):
    its, data = [], {k: [] for k in KEYS}
    for line in open(path):
        m = pat.search(line)
        if m:
            its.append(int(m[3]))
            for i, k in enumerate(KEYS):
                data[k].append(float(m[i + 4]))
    return np.array(its), {k: np.array(v, float) for k, v in data.items()}


def mavg(y, w=25):
    return y if len(y) < w else np.convolve(y, np.ones(w) / w, mode="same")


_PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(_PLOTS, exist_ok=True)
out = sys.argv[1]
if not os.path.isabs(out) and not out.startswith(("plots/", "./", "../")):
    out = os.path.join(_PLOTS, out)

runs = []
for arg in sys.argv[2:]:
    parts = arg.split(":")
    label, path = parts[0], parts[1]
    # optional 3rd field = samples_per_iter (for K-accum / multi-batch runs);
    # if given for all runs, x-axis becomes env-steps × 10⁶ instead of iter
    spi = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
    its, data = parse_log(path)
    runs.append((label, its, data, spi))

panels = [("KL",     "D_KL(π_θ ‖ π_ref)        [Fig A1 — left]"),
          ("r_h",    "Log-Likelihood  log π_ref(aₜ)   [Fig A1 — mid]"),
          ("H",      "Entropy  H(π_θ)          [Fig A1 — right]"),
          None,
          ("loss",   "Total loss   L = −L_PPO + β·D_KL   [Eq. 2]"),
          ("r_task", "r_task   (Variant 4 reward, ≤0)"),
          ("vL",     "PPO value loss"),
          ("g",      "grad-norm |g| pre-clip")]

# distinct, colour-blind-friendly palette for the runs
COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "k"]

use_env = all(r[3] > 0 for r in runs)         # env-step x-axis if all given
fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
for ax, p in zip(axes.flat, panels):
    if p is None:
        ax.set_visible(False)
        continue
    k, title = p
    for (label, its, data, spi), c in zip(runs, COLOURS):
        y = data[k]
        x = its * spi / 1e6 if use_env else its
        ax.plot(x, y, color=c, lw=1.4, label=label)         # raw per-iter
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("env-steps × 10⁶" if use_env else "iteration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)

fig.suptitle("SPACeR β-sweep overlay — Variant 4 (KL + r_inf), W=24, "
             "5,000-scene full resample · α=0   ·   "
             f"{len(runs)} runs × 1500 iters  (raw + 25-iter moving avg)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out, dpi=110)
print(f"wrote {out}")
