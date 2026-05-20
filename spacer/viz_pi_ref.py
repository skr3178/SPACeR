"""Static 2x2 PNG: GT log, pi_ref (CAT-K clsft_E9), pi_theta (random init),
and an overlay of all three on the same road graph.

Pipeline (1 GPUDrive validation scene, 1 world):
  A: GT  -> WOMD logged 8 s future (no GPU forward)
  B: pi_ref -> SMARTDecoder.eval() autoregressive 8 s (2 Hz token rollout)
  C: pi_theta -> fresh TokenPolicy stepping GPUDrive sim for 8 s (random init)
  D: overlay (A dashed, B solid, C dotted)

Run (inside spacer-dev):
  docker exec -i -e HF_HUB_OFFLINE=1 -w /catk spacer-dev \
      python /spacer/viz_pi_ref.py --out /tmp/viz_pi_ref.png
"""
import argparse
import dataclasses
import sys

sys.path.insert(0, "/catk")
sys.path.insert(0, "/spacer")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.datatypes.observation import GlobalEgoState

from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor

from gpudrive_to_smart import extract_gpudrive_scene, scene_dict_to_heterodata
from policy_token import TokenPolicy
from token_decode import decode_token_sequence

DEV = "cuda"
NUM_HIST = 11
NUM_FUT = 80
NUM_STEPS = NUM_HIST + NUM_FUT          # 91
SHIFT = 5


# --------------------------------------------------------------------------
# build + load
# --------------------------------------------------------------------------
def build_env(n_scenes=1):
    cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
    loader = SceneDataLoader(
        root="/gpd/data/processed/validation",
        batch_size=n_scenes, dataset_size=n_scenes,
        sample_with_replacement=False,
    )
    ec = dataclasses.replace(
        EnvConfig(), dynamics_model="state", collision_behavior="ignore",
        remove_non_vehicles=cfg.remove_non_vehicles, obs_radius=cfg.obs_radius,
    )
    return GPUDriveTorchEnv(
        config=ec, data_loader=loader,
        max_cont_agents=cfg.max_controlled_agents, device=DEV,
    )


def load_ref(ckpt="/ckpt/clsft_E9.ckpt"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = OmegaConf.create(ck["hyper_parameters"]).model_config
    tp = TokenProcessor(**mc.token_processor)
    dec = SMARTDecoder(**mc.decoder, n_token_agent=tp.n_token_agent).to(DEV).eval()
    dec.load_state_dict(
        {k[8:]: v for k, v in ck["state_dict"].items() if k.startswith("encoder.")},
        strict=True,
    )
    for p in dec.parameters():
        p.requires_grad_(False)
    return tp, dec


# --------------------------------------------------------------------------
# rollouts
# --------------------------------------------------------------------------
@torch.no_grad()
def pi_ref_rollout(env, tp, dec, world_idx=0):
    s = extract_gpudrive_scene(env, world_idx)
    hd = scene_dict_to_heterodata(s, "viz")
    b = Batch.from_data_list([hd])
    tmap, tag = tp(b)
    to_dev = lambda d: {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in d.items()}
    pred = dec(to_dev(tmap), to_dev(tag))
    traj = pred["pred_pos"].detach().cpu().numpy()       # [A_ref, T_tok, 2]
    pvalid = pred["pred_valid"].detach().cpu().numpy()
    ref_ids = b["agent"]["id"].numpy().astype(np.int64)
    return s, traj, pvalid, ref_ids


@torch.no_grad()
def pi_theta_rollout(env, tp, world_idx=0):
    """Untrained TokenPolicy closed-loop in GPUDrive; returns [A, 91, 2] @ 10 Hz."""
    obs = env.reset()
    cmask = env.cont_agent_mask                           # [W, A]
    s0 = extract_gpudrive_scene(env, world_idx)
    A = s0["pos_xy"].shape[0]

    odim = obs[cmask].shape[-1]
    policy = TokenPolicy(obs_dim=odim).to(DEV)
    ttraj = tp._get_agent_shape_and_token_traj(
        torch.zeros(A, dtype=torch.long))[2]              # [A, n_token, 4, 2]

    buf = np.zeros((A, NUM_STEPS, 2), np.float32)
    prev_pos = torch.tensor(s0["pos_xy"][:, 0], dtype=torch.float32)
    prev_head = torch.tensor(s0["yaw"][:, 0], dtype=torch.float32)
    steps = list(range(SHIFT, NUM_STEPS, SHIFT))          # 18 token-steps
    t = 0
    for i in steps:
        g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                       backend="torch", device=DEV)
        while t <= i and t < NUM_STEPS:
            buf[:, t, 0] = g.pos_x[0].cpu().numpy()
            buf[:, t, 1] = g.pos_y[0].cpu().numpy()
            t += 1
        x = obs[cmask]
        tok, _, _, _ = policy(x, deterministic=False)
        tok_all = torch.zeros(A, 1, dtype=torch.long)
        tok_all[cmask[0].cpu()] = tok.detach().cpu().view(-1, 1)
        # non-controlled agents -> token 0 (zero-motion primitive)
        dpos, dhead = decode_token_sequence(
            tok_all, prev_pos, prev_head, ttraj,
            torch.ones(A, 1, dtype=torch.bool))
        dpos, dhead = dpos[:, 0], dhead[:, 0]
        for _ in range(SHIFT):
            gg = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                            backend="torch", device=DEV)
            act = torch.zeros((cmask.shape[0], A, 10), dtype=torch.float32, device=DEV)
            act[0, :, 0] = dpos[:, 0].to(DEV)
            act[0, :, 1] = dpos[:, 1].to(DEV)
            act[0, :, 2] = gg.pos_z[0]
            act[0, :, 3] = dhead.to(DEV)
            env.step_dynamics(act)
        prev_pos, prev_head = dpos, dhead
        obs = env.get_obs()
    g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                   backend="torch", device=DEV)
    while t < NUM_STEPS:
        buf[:, t, 0] = g.pos_x[0].cpu().numpy()
        buf[:, t, 1] = g.pos_y[0].cpu().numpy()
        t += 1
    return buf            # [A_scene, 91, 2]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def draw_roadgraph(ax, s):
    keep = (s["rg_type"] >= 0) & (s["rg_id"] >= 0)
    rg_x, rg_y, rg_id = s["rg_x"][keep], s["rg_y"][keep], s["rg_id"][keep]
    for sid in np.unique(rg_id):
        m = rg_id == sid
        if m.sum() < 2:
            continue
        ax.plot(rg_x[m], rg_y[m], color="0.80", linewidth=0.6, zorder=1)


def plot_polyline(ax, xy, color, linestyle, linewidth=1.5, zorder=4, alpha=0.95):
    if len(xy) < 2:
        return
    ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.scatter([xy[0, 0]], [xy[0, 1]], color=color, s=20,
               edgecolor="black", linewidth=0.3, zorder=5)


def render_four(s, ref_traj, ref_valid, ref_ids, theta_traj, out_path):
    pos_gt = s["pos_xy"]                                 # [A, 91, 2]
    valid = s["valid"]                                   # [A, 91]
    cmask = s["cmask"]
    obj_id = s["obj_id"]
    id_to_scene = {int(i): k for k, i in enumerate(obj_id)}

    # match pi_ref agents to scene rows
    ref_to_scene = np.array([id_to_scene.get(int(rid), -1) for rid in ref_ids])
    keep = ref_to_scene >= 0
    ref_to_scene = ref_to_scene[keep]
    ref_traj = ref_traj[keep]
    ref_valid = ref_valid[keep]

    cmap = plt.get_cmap("tab20")
    # color per (scene-row) agent
    color_per_scene = {si: cmap((idx % 20)) for idx, si in enumerate(ref_to_scene)}

    # Shared extent from GT + π_ref only (π_θ may diverge wildly when untrained;
    # let its outliers fly off-screen rather than blow up the bounds).
    pts = []
    for j, si in enumerate(ref_to_scene):
        pts.append(ref_traj[j][ref_valid[j].astype(bool)])
        pts.append(pos_gt[si, NUM_HIST:][valid[si, NUM_HIST:]])
    pts = [p for p in pts if len(p)]
    all_pts = np.concatenate(pts, axis=0) if pts else np.zeros((1, 2))
    xmin, ymin = all_pts.min(0) - 15; xmax, ymax = all_pts.max(0) + 15

    fig, axes = plt.subplots(2, 2, figsize=(15, 14), dpi=120)
    (axA, axB), (axC, axD) = axes
    titles = {
        axA: "A. GT logged 8 s future (WOMD)",
        axB: "B. π_ref closed-loop (CAT-K clsft_E9)",
        axC: "C. π_θ closed-loop (fresh TokenPolicy, untrained — expected garbage)",
        axD: "D. Overlay (A dashed · B solid · C dotted)",
    }
    for ax in (axA, axB, axC, axD):
        draw_roadgraph(ax, s)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.grid(True, color="0.94", linewidth=0.4, zorder=0)
        ax.set_title(titles[ax], fontsize=10)

    drawn = 0
    for j, si in enumerate(ref_to_scene):
        col = color_per_scene[si]
        # A: GT future
        v_gt = valid[si, NUM_HIST:]
        if v_gt.sum() >= 2:
            pg = pos_gt[si, NUM_HIST:][v_gt]
            plot_polyline(axA, pg, col, "-", 1.5, 4)
            plot_polyline(axD, pg, col, "--", 1.3, 3, alpha=0.85)
        # B: pi_ref
        v = ref_valid[j].astype(bool)
        if v.sum() >= 2:
            pr = ref_traj[j][v]
            if not np.allclose(pr, 0):
                plot_polyline(axB, pr, col, "-", 1.7, 4)
                plot_polyline(axD, pr, col, "-", 1.6, 4, alpha=0.95)
        # C: pi_theta — only draw for agents valid at the rollout start
        if valid[si, NUM_HIST]:
            pt = theta_traj[si, NUM_HIST:]
            if len(pt) >= 2:
                plot_polyline(axC, pt, col, "-", 1.5, 4)
                plot_polyline(axD, pt, col, ":", 1.4, 3, alpha=0.85)
        drawn += 1

    fig.suptitle(
        f"GPUDrive validation scene · 8 s rollouts · agents drawn={drawn} · "
        f"controlled in env={int(cmask.sum())}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[viz] wrote {out_path}  ({drawn} agents)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/viz_pi_ref.png")
    ap.add_argument("--scene", type=int, default=0)
    args = ap.parse_args()

    print("[viz] building env (1 scene, validation)...")
    env = build_env(1); env.reset()

    print("[viz] loading π_ref (clsft_E9)...")
    tp, dec = load_ref()

    print("[viz] π_ref closed-loop (autoregressive, no env stepping)...")
    s, ref_traj, ref_valid, ref_ids = pi_ref_rollout(env, tp, dec, args.scene)
    print(f"[viz]   ref_traj={ref_traj.shape}  ref agents={len(ref_ids)}")

    print("[viz] π_θ closed-loop (random init, steps env 80x)...")
    theta_traj = pi_theta_rollout(env, tp, args.scene)
    print(f"[viz]   theta_traj={theta_traj.shape}")

    render_four(s, ref_traj, ref_valid, ref_ids, theta_traj, args.out)


if __name__ == "__main__":
    main()
