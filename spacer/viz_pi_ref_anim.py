"""Animated GIF: 3-panel side-by-side comparison of
  A. GT logged 8 s future (WOMD)
  B. π_ref closed-loop (CAT-K clsft_E9)
  C. π_θ closed-loop (fresh untrained TokenPolicy)
on one GPUDrive validation scene at 10 Hz (80 frames, 8 s).

π_ref is autoregressive 2 Hz; linearly interpolated to 10 Hz between token-step
endpoints for visual smoothness. π_θ steps GPUDrive 80x at 10 Hz directly.
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
from matplotlib.animation import FuncAnimation, PillowWriter
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
NUM_STEPS = NUM_HIST + NUM_FUT
SHIFT = 5
TRAIL = 20


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


@torch.no_grad()
def pi_ref_rollout(env, tp, dec, world_idx=0):
    s = extract_gpudrive_scene(env, world_idx)
    hd = scene_dict_to_heterodata(s, "viz")
    b = Batch.from_data_list([hd])
    tmap, tag = tp(b)
    to_dev = lambda d: {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in d.items()}
    pred = dec(to_dev(tmap), to_dev(tag))
    traj = pred["pred_pos"].detach().cpu().numpy()
    pvalid = pred["pred_valid"].detach().cpu().numpy()
    ref_ids = b["agent"]["id"].numpy().astype(np.int64)
    return s, traj, pvalid, ref_ids


def interp_to_10hz(token_pos, token_valid):
    A, T_tok, _ = token_pos.shape
    T_fut_tok = T_tok - 1
    T_10 = T_fut_tok * SHIFT
    out = np.zeros((A, T_10, 2), dtype=np.float32)
    out_valid = np.zeros((A, T_10), dtype=bool)
    for a in range(A):
        prev_pos = token_pos[a, 0]; prev_val = bool(token_valid[a, 0])
        for k in range(1, T_tok):
            cur_pos = token_pos[a, k]; cur_val = bool(token_valid[a, k])
            base = (k - 1) * SHIFT
            if prev_val and cur_val:
                for f in range(SHIFT):
                    t = (f + 1) / SHIFT
                    out[a, base + f] = prev_pos + t * (cur_pos - prev_pos)
                    out_valid[a, base + f] = True
            prev_pos, prev_val = cur_pos, cur_val
    return out[:, :NUM_FUT], out_valid[:, :NUM_FUT]


@torch.no_grad()
def pi_theta_rollout(env, tp, world_idx=0):
    obs = env.reset()
    cmask = env.cont_agent_mask
    s0 = extract_gpudrive_scene(env, world_idx)
    A = s0["pos_xy"].shape[0]
    odim = obs[cmask].shape[-1]
    policy = TokenPolicy(obs_dim=odim).to(DEV)
    ttraj = tp._get_agent_shape_and_token_traj(
        torch.zeros(A, dtype=torch.long))[2]
    buf = np.zeros((A, NUM_STEPS, 2), np.float32)
    prev_pos = torch.tensor(s0["pos_xy"][:, 0], dtype=torch.float32)
    prev_head = torch.tensor(s0["yaw"][:, 0], dtype=torch.float32)
    steps = list(range(SHIFT, NUM_STEPS, SHIFT))
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
    return buf


def draw_static(ax, s, title):
    keep = (s["rg_type"] >= 0) & (s["rg_id"] >= 0)
    rg_x, rg_y, rg_id = s["rg_x"][keep], s["rg_y"][keep], s["rg_id"][keep]
    for sid in np.unique(rg_id):
        m = rg_id == sid
        if m.sum() < 2:
            continue
        ax.plot(rg_x[m], rg_y[m], color="0.82", linewidth=0.6, zorder=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="0.94", linewidth=0.4, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/viz_pi_ref.gif")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    print("[anim] env + π_ref...")
    env = build_env(1); env.reset()
    tp, dec = load_ref()

    print("[anim] π_ref rollout...")
    s, traj_tok, pvalid, ref_ids = pi_ref_rollout(env, tp, dec, args.scene)
    print(f"[anim]   shape={traj_tok.shape}")
    ref_10, ref_v10 = interp_to_10hz(traj_tok, pvalid)            # [A_ref, 80, 2]

    print("[anim] π_θ rollout (steps env 80x)...")
    theta_traj = pi_theta_rollout(env, tp, args.scene)            # [A_scene, 91, 2]
    theta_fut = theta_traj[:, NUM_HIST:NUM_HIST + NUM_FUT]        # [A_scene, 80, 2]

    pos_gt = s["pos_xy"][:, NUM_HIST:NUM_HIST + NUM_FUT]
    gt_v = s["valid"][:, NUM_HIST:NUM_HIST + NUM_FUT]
    obj_id = s["obj_id"]; cmask = s["cmask"]
    id_to_scene = {int(i): k for k, i in enumerate(obj_id)}
    ref_to_scene = np.array([id_to_scene.get(int(rid), -1) for rid in ref_ids])
    keep_ref = ref_to_scene >= 0
    ref_ids = ref_ids[keep_ref]
    ref_10 = ref_10[keep_ref]
    ref_v10 = ref_v10[keep_ref]
    ref_to_scene = ref_to_scene[keep_ref]

    # Extent from GT + π_ref only (let π_θ outliers fly off-screen if untrained)
    pts = []
    for j, si in enumerate(ref_to_scene):
        pts.append(ref_10[j][ref_v10[j]])
        pts.append(pos_gt[si][gt_v[si]])
    pts = [p for p in pts if len(p)]
    all_pts = np.concatenate(pts, axis=0) if pts else np.zeros((1, 2))
    xmin, ymin = all_pts.min(0) - 15; xmax, ymax = all_pts.max(0) + 15

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 7), dpi=100)
    draw_static(axA, s, "A. GT logged future (WOMD)")
    draw_static(axB, s, "B. π_ref closed-loop (CAT-K)")
    draw_static(axC, s, "C. π_θ closed-loop (untrained)")
    for ax in (axA, axB, axC):
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    cmap = plt.get_cmap("tab20")
    n_ref = len(ref_ids)
    colors = [cmap(j % 20) for j in range(n_ref)]
    dotsA = axA.scatter([], [], s=40, edgecolor="black", linewidth=0.4, zorder=5)
    dotsB = axB.scatter([], [], s=40, edgecolor="black", linewidth=0.4, zorder=5)
    dotsC = axC.scatter([], [], s=40, edgecolor="black", linewidth=0.4, zorder=5)
    trailsA = [axA.plot([], [], color=c, linewidth=1.4, alpha=0.9, zorder=4)[0]
               for c in colors]
    trailsB = [axB.plot([], [], color=c, linewidth=1.4, alpha=0.9, zorder=4)[0]
               for c in colors]
    trailsC = [axC.plot([], [], color=c, linewidth=1.4, alpha=0.9, zorder=4)[0]
               for c in colors]
    title_sup = fig.suptitle("", fontsize=11)

    def update(f):
        xyA, cA, xyB, cB, xyC, cC = [], [], [], [], [], []
        for j, si in enumerate(ref_to_scene):
            # A: GT
            if gt_v[si, f]:
                xyA.append(pos_gt[si, f]); cA.append(colors[j])
                lo = max(0, f - TRAIL)
                m = gt_v[si, lo:f + 1]
                tr = pos_gt[si, lo:f + 1][m]
                if len(tr): trailsA[j].set_data(tr[:, 0], tr[:, 1])
            else:
                trailsA[j].set_data([], [])
            # B: pi_ref
            if ref_v10[j, f]:
                xyB.append(ref_10[j, f]); cB.append(colors[j])
                lo = max(0, f - TRAIL)
                m = ref_v10[j, lo:f + 1]
                tr = ref_10[j, lo:f + 1][m]
                if len(tr): trailsB[j].set_data(tr[:, 0], tr[:, 1])
            else:
                trailsB[j].set_data([], [])
            # C: pi_theta
            xyC.append(theta_fut[si, f]); cC.append(colors[j])
            lo = max(0, f - TRAIL)
            tr = theta_fut[si, lo:f + 1]
            if len(tr) >= 2: trailsC[j].set_data(tr[:, 0], tr[:, 1])
        dotsA.set_offsets(np.array(xyA) if xyA else np.empty((0, 2)))
        dotsA.set_facecolor(cA if cA else "none")
        dotsB.set_offsets(np.array(xyB) if xyB else np.empty((0, 2)))
        dotsB.set_facecolor(cB if cB else "none")
        dotsC.set_offsets(np.array(xyC) if xyC else np.empty((0, 2)))
        dotsC.set_facecolor(cC if cC else "none")
        t_s = (f + 1) * 0.1
        title_sup.set_text(f"scene {args.scene}  ·  t = {t_s:.1f} s / 8.0 s  ·  "
                           f"tracked={n_ref}  ·  controlled in env={int(cmask.sum())}")
        return (dotsA, dotsB, dotsC, *trailsA, *trailsB, *trailsC, title_sup)

    anim = FuncAnimation(fig, update, frames=NUM_FUT, blit=False)
    print(f"[anim] saving {args.out} @ {args.fps} fps ({NUM_FUT} frames)...")
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"[anim] wrote {args.out}")


if __name__ == "__main__":
    main()
