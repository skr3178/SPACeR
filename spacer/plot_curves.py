"""Plot SPACeR training curves from a run log.

    python3 spacer/plot_curves.py <log> [<out.png>]   (run on host)

Default output → the repo-root `plots/` folder (images kept out of logs/).

Layout mirrors the paper's Figure A1:
  ROW 1 — the three Fig A1 panels: D_KL, Log-Likelihood, Entropy.
  ROW 2 — our PPO / Variant-4 diagnostics: r_task, value loss, |g|.
Paper Fig A1 is a *sweep* (a row per α and per β); we run one configuration,
so this shows that single point — α and β are annotated in the title.

Parses the per-iter `[β=… W=…] itNNN: r_task=… r_h=… KL=… H=… pg=… vL=…
loss=… |g|=…` lines (raw + moving average). Re-runnable any time.
"""
import sys, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1]
# default output → the repo-root plots/ folder (kept separate from logs/)
_PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(_PLOTS, exist_ok=True)
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_PLOTS,
                                                         "training_curves.png")

pat = re.compile(
    r"\[(?:ABL )?β=([\d.]+) W=(\d+)\] it(\d+): "
    r"r_task=([-+\d.]+) r_h=([-+\d.]+) KL=([-+\d.]+) "
    r"H=([-+\d.]+) pg=([-+\d.]+) vL=([-+\d.]+) loss=([-+\d.]+) \|g\|=([-+\d.]+)")

keys = ["r_task", "r_h", "KL", "H", "pg", "vL", "loss", "g"]
its, data = [], {k: [] for k in keys}
beta = world = None
for line in open(log):
    m = pat.search(line)
    if m:
        beta, world = m[1], m[2]
        its.append(int(m[3]))
        for i, k in enumerate(keys):
            data[k].append(float(m[i + 4]))
its = np.array(its)
# α is not in the log line — Variant 4 fixes it at 0 (run() default alpha=0.0).
ALPHA = "0"


def mavg(y, w=25):
    y = np.asarray(y, float)
    return y if len(y) < w else np.convolve(y, np.ones(w) / w, mode="same")


# row 1 = paper Fig A1 trio ; row 2 = our PPO / Variant-4 diagnostics
# (incl. the total optimised loss, Eq. 2). None ⇒ blank/hidden cell.
panels = [("KL",     "D_KL(π_θ ‖ π_ref)        [Fig A1 — left]",     "tab:blue"),
          ("r_h",    "Log-Likelihood  log π_ref(aₜ)   [Fig A1 — mid]", "tab:purple"),
          ("H",      "Entropy  H(π_θ)          [Fig A1 — right]",    "tab:green"),
          None,
          ("loss",   "Total loss   L = −L_PPO + β·D_KL   [Eq. 2]",   "tab:cyan"),
          ("r_task", "r_task   [ours — Variant 4 reward, ≤0]",       "tab:red"),
          ("vL",     "PPO value loss   [ours]",                      "tab:orange"),
          ("g",      "grad-norm |g| pre-clip   [ours]",              "tab:gray")]

fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
for ax, p in zip(axes.flat, panels):
    if p is None:
        ax.set_visible(False)
        continue
    k, title, c = p
    y = data[k]
    ax.plot(its, y, color=c, lw=0.5, alpha=0.35)
    ax.plot(its, mavg(y), color=c, lw=1.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)

fig.suptitle(
    f"SPACeR long run — Variant 4 (KL + r_inf)   ·   "
    f"α = {ALPHA}   β = {beta}   ·   W = {world}   ·   {len(its)} iters\n"
    f"row 1 = paper Figure A1 panels   ·   row 2 = our PPO / Variant-4 "
    f"diagnostics incl. total loss   (raw + 25-iter moving avg)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out, dpi=110)
print(f"wrote {out}  ({len(its)} iters, α={ALPHA} β={beta} W={world})")
