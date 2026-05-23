"""Visualise one raw GPUDrive scene file (no sim / no CUDA needed).

    python3 spacer/viz_scene.py <scene.json> [<out.png>]   (run on host)

Default output → the repo-root `plots/` folder (images kept out of logs/).
Scene files: /home/skr/gpudrive_data/{training/group_0,validation}/*.json

Parses a single GPUDrive-processed WOMD scene JSON and renders a top-down map:
road elements (coloured by type), every agent's logged trajectory, each agent's
initial bounding box (oriented by heading), and the ego goal. The ego / self-
driving car (metadata.sdc_track_index) is drawn in red.
"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

scene = sys.argv[1]
# default output → the repo-root plots/ folder (kept separate from logs/)
_PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(_PLOTS, exist_ok=True)
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_PLOTS,
                                                         "scene_viz.png")
d = json.load(open(scene))

# --- roads: one polyline per element, coloured / styled by type ---
ROAD_STYLE = {            # type        -> (colour, linewidth, linestyle)
    "lane":       ("0.82", 0.8, "-"),
    "road_edge":  ("0.10", 1.4, "-"),
    "road_line":  ("0.55", 0.8, "--"),
    "crosswalk":  ("tab:blue", 1.2, "-"),
    "speed_bump": ("tab:orange", 1.2, "-"),
    "driveway":   ("tan", 0.8, "-"),
    "stop_sign":  ("tab:red", 0.0, "-"),     # rendered as a point
}
fig, ax = plt.subplots(figsize=(12, 12))
for r in d["roads"]:
    g = r.get("geometry", [])
    if not g:
        continue
    xs = [p["x"] for p in g]
    ys = [p["y"] for p in g]
    c, lw, ls = ROAD_STYLE.get(r["type"], ("0.7", 0.6, "-"))
    if r["type"] == "stop_sign":
        ax.plot(xs, ys, "o", color=c, ms=7, zorder=2)
    else:
        ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=1)

# --- agents: logged trajectory + initial oriented bounding box + ego goal ---
sdc = d.get("metadata", {}).get("sdc_track_index", -1)
TYPE_COL = {"vehicle": "tab:green", "pedestrian": "tab:purple",
            "cyclist": "tab:orange"}
n_drawn = 0
for i, o in enumerate(d["objects"]):
    valid = np.array(o["valid"], bool)
    if not valid.any():
        continue
    n_drawn += 1
    pos = np.array([[p["x"], p["y"]] for p in o["position"]])
    is_ego = (i == sdc)
    col = "red" if is_ego else TYPE_COL.get(o["type"], "0.4")
    # logged trajectory over the valid steps
    ax.plot(pos[valid, 0], pos[valid, 1], color=col,
            lw=1.8 if is_ego else 0.9, alpha=0.95 if is_ego else 0.55, zorder=3)
    # initial oriented bounding box (rear-left corner, rotated by heading)
    t0 = int(np.argmax(valid))
    x0, y0 = pos[t0]
    h = o["heading"][t0]
    L, W = o["length"], o["width"]
    cos, sin = np.cos(h), np.sin(h)
    corner = (x0 - (L / 2) * cos + (W / 2) * sin,
              y0 - (L / 2) * sin - (W / 2) * cos)
    ax.add_patch(Rectangle(corner, L, W, angle=np.degrees(h),
                           facecolor=col, edgecolor="black", lw=0.8,
                           alpha=0.9, zorder=4))
    # ego goal
    gp = o.get("goalPosition")
    if gp and is_ego:
        ax.plot(gp["x"], gp["y"], "*", color="red", ms=22,
                markeredgecolor="black", zorder=5)

legend = [Line2D([0], [0], color="red", lw=2, label="ego (SDC)"),
          Line2D([0], [0], color="tab:green", lw=2, label="vehicle"),
          Line2D([0], [0], color="tab:purple", lw=2, label="pedestrian"),
          Line2D([0], [0], color="tab:orange", lw=2, label="cyclist"),
          Line2D([0], [0], color="0.10", lw=1.4, label="road edge"),
          Line2D([0], [0], color="0.55", lw=0.8, ls="--", label="road line"),
          Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                 markeredgecolor="black", ms=14, label="ego goal")]
ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
ax.set_aspect("equal")
ax.set_title(f"GPUDrive scene  {d.get('scenario_id', '?')}   "
             f"({n_drawn} valid agents, {len(d['roads'])} road elements)\n"
             f"{scene.split('/')[-1]}", fontsize=10)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f"wrote {out}  (scenario {d.get('scenario_id','?')}, "
      f"{n_drawn} agents, {len(d['roads'])} roads)")
