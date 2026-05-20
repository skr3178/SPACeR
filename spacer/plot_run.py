"""Parse a train_spacer log and plot all per-iter terms.

Usage:
  python plot_run.py <log_file> [--out fig.png]
"""
import argparse, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(
    r"it(\d+):\s+r_task=([-+]?\d+\.\d+)\s+r_h=([-+]?\d+\.\d+)\s+"
    r"KL=([-+]?\d+\.\d+)\s+loss=([-+]?\d+\.\d+)\s+\|g\|=([-+]?\d+\.\d+)")


def parse(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                rows.append([float(x) if i else int(x)
                             for i, x in enumerate(m.groups())])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        sys.exit(f"No iter lines parsed from {args.log}")
    it, rt, rh, kl, ls, gn = zip(*rows)
    out = args.out or args.log.replace(".log", ".png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    panels = [
        ("KL  (Eq.5)",          kl, "tab:blue"),
        ("r_task  (Eq.1 r_inf)", rt, "tab:red"),
        ("r_h  (Eq.3 LLH)",     rh, "tab:green"),
        ("loss  = -L_PPO + β·KL", ls, "tab:purple"),
        ("|g|  (grad norm)",    gn, "tab:orange"),
    ]
    for ax, (title, y, c) in zip(axes.flat, panels):
        ax.plot(it, y, color=c, lw=1.2)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
    axes[1, 2].axis("off")
    for ax in axes[1, :]:
        ax.set_xlabel("iteration")

    # Header
    fig.suptitle(f"{args.log.split('/')[-1]}   ({len(rows)} iters)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
