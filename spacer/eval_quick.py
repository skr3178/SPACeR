"""Phase-A quick eval (Eval_Plan.md §Phase A).

Closed-loop π_θ rollouts in GPUDrive on validation scenes. Reports the
GPUDrive-derivable subset of the paper's Table 1 columns plus the Fig A1
training-dynamics signals:

  paper Table 1 column   →  here
  --------------------------------------------------
  Collision ↓            →  collision_rate
  Off-road ↓             →  off_road_rate
  minADE ↓               →  min_ade_m   (min over K rollouts, frame-correct)
  Throughput ↑           →  throughput_scenarios_per_s
  (Composite/Kinematic/Interactive/Map need the WOSAC library — Phase D)

  Fig A1 panels (training dynamics, NOT in Table 1):
  D_KL → kl_mean   Log-Likelihood → r_h_mean   Entropy → entropy_mean

Up to three arms, evaluated on **identical scenes** (interleaved per
scene-batch — the env cannot be rebuilt in-process because Madrona/CUDA cannot
re-init):
  - trained : π_θ loaded from --ckpt
  - random  : freshly-init TokenPolicy (sanity baseline)
  - ref     : π_ref (CAT-K clsft_E9) closed-loop, via --ref-arm. This is the
              Phase A.5 reference baseline — π_ref's own dec.inference()
              rollout drives GPUDrive (ref agents) with GT log-replay for the
              rest. kl_mean is N/A for this arm (no π_θ); r_h_mean/entropy_mean
              come from π_ref's teacher-forced distribution.

minADE = per-agent min over K rollouts, then mean over agents — matches the
paper's "minimum displacement error over rollouts" definition. K is --rollouts.

CLI
---
  python eval_quick.py --ckpt PATH [--scene-batches N] [--worlds W]
                       [--rollouts K] [--ref-arm] [--out PATH]

  total scenes evaluated = scene-batches × worlds
  total rollouts per arm = scene-batches × worlds × rollouts

No new deps; runs inside spacer-dev. Output JSON →
/spacer/eval_runs/<ckpt_stem>/quick_metrics.json.
"""
import os, sys, time, json, argparse, math
import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

sys.path.insert(0, "/spacer")
from train_spacer import (build_env, rollout, score_ref, load_ref, set_state,
                          TokenPolicy, load_ckpt, REF_STEP_OFFSET, NUM_STEPS, DEV)
from anchor import r_humanlike, kl_theta_ref, align_agents
from gpudrive_to_smart import extract_gpudrive_scene, scene_dict_to_heterodata
from gpudrive.datatypes.info import Info
from gpudrive.datatypes.observation import GlobalEgoState

NUM_HIST = 11                  # WOMD history steps (step 10 = "current")
# CAT-K's own closed-loop validation rollout sampling (smart.yaml
# `validation_rollout_sampling`): top-k by probability, k=5.
REF_SAMPLING = OmegaConf.create({"criterium": "topk_prob", "num_k": 5,
                                 "temp": 1.0})


# ---------- metric helpers ---------------------------------------------------

def _flag_rate(env, cmask, flag_name):
    """Per-agent fraction with `flag_name` set at the current env state."""
    info = Info.from_tensor(env.sim.info_tensor(), backend="torch", device=DEV)
    v = getattr(info, flag_name)
    fired = v[cmask].bool()
    return float(fired.float().mean()) if fired.numel() else float("nan")


SENTINEL_THRESH = 5000.0   # |pos| above this ⇒ GPUDrive off-map sentinel
                           # (real mean-centered positions are O(100 m);
                           #  off-road agents are parked at ~−11000)
FULL_COV = 0.999           # coverage ≥ this ⇒ agent stayed on-map for its
                           # whole GT-valid window in that rollout


def _ade_per_agent(rolled_xy, gt_xy, agent_valid, controlled):
    """Per-agent ADE = mean_t ||rolled − gt|| over steps where GT is valid
    AND the rolled position is on-map (not an off-road sentinel).

    rolled_xy / gt_xy : [A, T, 2] (BOTH in the same coord frame)
    agent_valid       : [A, T] bool — GT validity
    controlled        : [A]    bool — cont_agent_mask for this world

    **Controlled-agents only.** Non-controlled agents are log-replayed by
    GPUDrive (`rolled == gt` by construction → ADE 0); including them swamps
    the average with trivial zeros. `has` is True only for controlled agents.

    Off-road handling: when a controlled agent goes off-road GPUDrive parks
    it at the ~−11000 sentinel. Those steps are masked out (they would inject
    a ~15 km error); the departure itself is captured by off_road_rate, and
    `coverage` reports how much of the GT-valid window stayed on-map.

    Returns (ade[A] float32, has[A] bool, coverage[A] float32):
      has      = controlled AND has ≥1 scored step
      coverage = fraction of GT-valid steps with an on-map rolled position
                 (1.0 = on-map the whole window; low = left early)
    """
    d = np.linalg.norm(rolled_xy - gt_xy, axis=-1)           # [A, T]
    gt_m = agent_valid.astype(bool)                          # [A, T]
    on_map = (np.abs(rolled_xy) < SENTINEL_THRESH).all(axis=-1)   # [A, T]
    ctrl = controlled.astype(bool)                           # [A]
    scored = gt_m & on_map                                   # [A, T]
    has = scored.any(axis=1) & ctrl                          # [A]
    per_agent = (np.where(scored, d, 0.0).sum(axis=1)
                 / np.maximum(scored.sum(axis=1), 1))
    coverage = scored.sum(axis=1) / np.maximum(gt_m.sum(axis=1), 1)
    return (per_agent.astype(np.float32), has, coverage.astype(np.float32))


def _anchor_signals(rec, s_live_per_w, theta_ids_per_w, nc_per_w, tp, dec,
                    policy):
    """r_h, KL, H over the rollout — same recipe as spacer_iteration, so
    eval-time numbers are directly comparable to training-time numbers."""
    per_world = score_ref(tp, dec, s_live_per_w)
    off = REF_STEP_OFFSET
    nc_offsets = [0]
    for n in nc_per_w:
        nc_offsets.append(nc_offsets[-1] + n)
    rh_list, kl_list, ent_list = [], [], []
    for w, (ref_logits, exec_tok, vmask, ref_ids) in enumerate(per_world):
        ref_logits = ref_logits.detach()
        if ref_logits.numel() == 0 or nc_per_w[w] == 0:
            continue
        _, rh_w = r_humanlike(ref_logits, exec_tok, vmask)
        gather, amatch = align_agents(ref_ids, theta_ids_per_w[w])
        if not amatch.any():
            continue
        A_ref, T_ref, _ = ref_logits.shape
        s, e = nc_offsets[w], nc_offsets[w + 1]
        th_steps = []
        for j in range(T_ref):
            k = j + off
            obs_k = rec["obs"][k][s:e] if k < len(rec["obs"]) \
                else rec["obs"][-1][s:e]
            th_steps.append(policy.logits(obs_k))
        th = torch.stack(th_steps, dim=1)
        th_aligned = th[gather]
        kmask = vmask & amatch.unsqueeze(1)
        _, kl_w = kl_theta_ref(th_aligned, ref_logits, kmask)
        logp_th = torch.log_softmax(th_aligned, dim=-1)
        ent_per = -(logp_th.exp() * logp_th).sum(-1)
        ent_w = ent_per[kmask].mean() if kmask.any() \
                else torch.zeros((), device=DEV)
        rh_list.append(rh_w); kl_list.append(kl_w); ent_list.append(ent_w)
    if not rh_list:
        return float("nan"), float("nan"), float("nan")
    return (float(torch.stack(rh_list).mean()),
            float(torch.stack(kl_list).mean()),
            float(torch.stack(ent_list).mean()))


def _nanmean(xs):
    a = np.array([x for x in xs
                  if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(a.mean()) if a.size else float("nan")


# ---------- per-arm accumulation over one scene-batch ------------------------

def eval_batch(env, policy, tp, dec, gt_scenes, n_rollouts):
    """Run `n_rollouts` rollouts on the *current* scene-batch (already loaded
    via swap_data_batch). gt_scenes = per-world GT dicts, extracted once.

    minADE: per-agent ADE collected for each rollout, then min over rollouts.
    collision/off-road/goal: averaged over rollouts.
    anchor signals (r_h/KL/H): computed once (rollout 0) — stable enough and
    score_ref is the expensive call.

    Returns a dict of per-batch aggregates + the rollout wall-time.
    """
    cmask = env.cont_agent_mask
    W = cmask.shape[0]
    coll, offr, goal, rtask = [], [], [], []
    # per-rollout list of {world: (ade[A], has[A], coverage[A])}
    ade_runs = []
    rh = kl = ent = float("nan")
    t0 = time.time()
    for k in range(n_rollouts):
        rec, s_live, theta_ids, nc_per_w = rollout(env, policy)
        coll.append(_flag_rate(env, cmask, "collided"))
        offr.append(_flag_rate(env, cmask, "off_road"))
        goal.append(_flag_rate(env, cmask, "goal_achieved"))
        rtask.append(float(torch.stack([r.mean() for r in rec["rtask"]]).mean()))
        # ADE per world — both rolled and GT are now in the sim-native
        # (mean-centered) frame (extract_gpudrive_scene no longer restores
        # the mean), so they are directly comparable.
        run = {}
        for w, sl in enumerate(s_live):
            rolled = sl["pos_xy"]                       # [A, T, 2] sim-native
            gt = gt_scenes[w]["pos_xy"]                 # [A, T, 2] sim-native
            valid = gt_scenes[w]["valid"]
            ctrl = gt_scenes[w]["cmask"]                # [A] controlled mask
            T = min(rolled.shape[1], gt.shape[1], valid.shape[1])
            ade, has, cov = _ade_per_agent(rolled[:, :T], gt[:, :T],
                                           valid[:, :T], ctrl)
            run[w] = (ade, has, cov)
        ade_runs.append(run)
        if k == 0:
            rh, kl, ent = _anchor_signals(rec, s_live, theta_ids, nc_per_w,
                                          tp, dec, policy)
    dt = time.time() - t0
    # minADE: per controlled (world, agent), min ADE over the rollouts where
    # the agent stayed on-map for its WHOLE GT-valid window (coverage ≥
    # FULL_COV). This kills the degenerate-rollout bias — without it, `min`
    # would preferentially pick rollouts where the agent left the road early
    # (fewer, earlier steps scored ⇒ artificially small ADE).
    # An agent with NO full-coverage rollout is excluded from minADE and
    # counted in the completion rate instead.
    min_ades = []
    n_ctrl = 0          # controlled agents with data in ≥1 rollout
    n_completed = 0     # controlled agents with ≥1 full-coverage rollout
    for w in range(W):
        A = ade_runs[0][w][0].shape[0]
        for a in range(A):
            if not any(ade_runs[k][w][1][a] for k in range(n_rollouts)):
                continue                                  # not controlled
            n_ctrl += 1
            full = [ade_runs[k][w][0][a] for k in range(n_rollouts)
                    if ade_runs[k][w][1][a]
                    and ade_runs[k][w][2][a] >= FULL_COV]
            if full:
                n_completed += 1
                min_ades.append(float(min(full)))
    min_ade_mean = float(np.mean(min_ades)) if min_ades else float("nan")
    completion = (n_completed / n_ctrl) if n_ctrl else float("nan")
    return dict(collision_rate=_nanmean(coll),
                off_road_rate=_nanmean(offr),
                goal_rate=_nanmean(goal),
                r_task_mean=_nanmean(rtask),
                min_ade_m=min_ade_mean,
                ade_completion_rate=completion,
                r_h_mean=rh, kl_mean=kl, entropy_mean=ent,
                rollout_time_s=dt)


# ---------- π_ref arm (Phase A.5) -------------------------------------------

@torch.no_grad()
def ref_rollout(env, tp, dec):
    """Closed-loop π_ref rollout across all W worlds.

    π_ref's own dec.inference() produces a free closed-loop trajectory
    (pred_traj_10hz) per agent; GPUDrive is then driven to it (ref agents) with
    GT log-replay for every other agent, so collision/off-road flags reflect
    π_ref's driving in a realistic scene. Mirrors train_spacer.rollout()'s
    s_live contract so _ade_per_agent / _flag_rate work unchanged.

    Returns (s_live_per_w, r_h_mean, entropy_mean):
      s_live_per_w : list[dict] — per-world rolled scene (pos_xy [A,91,2])
      r_h_mean     : mean log π_ref(GT token)  (teacher-forced; = −NLL)
      entropy_mean : mean entropy of π_ref's teacher-forced distribution
    """
    env.reset()
    cmask = env.cont_agent_mask
    W = cmask.shape[0]
    scenes0 = [extract_gpudrive_scene(env, w) for w in range(W)]
    A_per_w = [s["pos_xy"].shape[0] for s in scenes0]
    to_dev = lambda d: {k: (v.to(DEV) if torch.is_tensor(v) else v)
                        for k, v in d.items()}

    # per-world π_ref inference + teacher-forced r_h / entropy
    ref_rows_w, roll_w, rollh_w = [], [], []
    rh_list, ent_list = [], []
    for w in range(W):
        b = Batch.from_data_list([scene_dict_to_heterodata(scenes0[w], f"ref_{w}")])
        tmap, tag = tp(b)
        tmap, tag = to_dev(tmap), to_dev(tag)
        # GPUDrive scenes are 2D — our tokenizer drops height, but SMART's
        # inference path reads gt_z_raw only to assemble pred_z_10hz (the WOSAC
        # height channel, unused by our xy-based metrics). Zero-fill it.
        if "gt_z_raw" not in tag:
            tag["gt_z_raw"] = torch.zeros(tag["gt_pos_raw"].shape[0], device=DEV)
        # teacher-forced forward → r_h (= log π_ref of GT) + entropy
        ptf = dec(tmap, tag)
        nvalid = ptf["next_token_valid"].bool()
        if nvalid.any():
            logp = torch.log_softmax(ptf["next_token_logits"].float(), -1)
            gt_tok = tag["gt_idx"][:, 2:].long()
            rh = logp.gather(-1, gt_tok.unsqueeze(-1)).squeeze(-1)[nvalid]
            ent = -(logp.exp() * logp).sum(-1)[nvalid]
            rh_list.append(rh.cpu()); ent_list.append(ent.cpu())
        # free closed-loop rollout → pred_traj_10hz [A_ref, 80, 2]
        pi = dec.inference(tmap, tag, REF_SAMPLING)
        roll = pi["pred_traj_10hz"].cpu().numpy()
        rollh = pi["pred_head_10hz"].cpu().numpy()
        ref_ids = b["agent"]["id"].cpu().numpy().astype(np.int64)
        id2row = {int(i): k for k, i in enumerate(scenes0[w]["obj_id"])}
        rows = np.array([id2row.get(int(r), -1) for r in ref_ids])
        nz = ~np.all(roll.reshape(len(roll), -1) == 0, axis=1)
        keep = (rows >= 0) & nz
        ref_rows_w.append(rows[keep])
        roll_w.append(roll[keep]); rollh_w.append(rollh[keep])

    # drive GPUDrive: 11 history steps = GT; 80 future steps = π_ref (ref
    # agents) + GT log-replay (others). Record realized poses into buf.
    buf_pos = [np.zeros((A_per_w[w], NUM_STEPS, 2), np.float32) for w in range(W)]
    buf_head = [np.zeros((A_per_w[w], NUM_STEPS), np.float32) for w in range(W)]
    for w in range(W):
        buf_pos[w][:, :NUM_HIST] = scenes0[w]["pos_xy"][:, :NUM_HIST]
        buf_head[w][:, :NUM_HIST] = scenes0[w]["yaw"][:, :NUM_HIST]
    n_fut = NUM_STEPS - NUM_HIST                              # 80
    for f in range(n_fut):
        step = NUM_HIST + f
        dpos_per_w, dhead_per_w = [], []
        for w in range(W):
            pos = scenes0[w]["pos_xy"][:, step].copy()        # GT log-replay
            head = scenes0[w]["yaw"][:, step].copy()
            if len(ref_rows_w[w]):
                fi = min(f, roll_w[w].shape[1] - 1)
                pos[ref_rows_w[w]] = roll_w[w][:, fi]         # π_ref agents
                head[ref_rows_w[w]] = rollh_w[w][:, fi]
            dpos_per_w.append(torch.tensor(pos, dtype=torch.float32))
            dhead_per_w.append(torch.tensor(head, dtype=torch.float32))
        set_state(env, dpos_per_w, dhead_per_w)
        g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                       backend="torch", device=DEV)
        for w in range(W):
            Aw = A_per_w[w]
            buf_pos[w][:, step, 0] = g.pos_x[w, :Aw].cpu().numpy()
            buf_pos[w][:, step, 1] = g.pos_y[w, :Aw].cpu().numpy()
            buf_head[w][:, step] = g.rotation_angle[w, :Aw].cpu().numpy()

    s_live_per_w = []
    for w in range(W):
        sl = dict(scenes0[w])
        sl["pos_xy"] = buf_pos[w]; sl["yaw"] = buf_head[w]
        s_live_per_w.append(sl)
    rh_mean = float(torch.cat(rh_list).mean()) if rh_list else float("nan")
    ent_mean = float(torch.cat(ent_list).mean()) if ent_list else float("nan")
    return s_live_per_w, rh_mean, ent_mean


def eval_batch_ref(env, tp, dec, gt_scenes, n_rollouts):
    """π_ref ref-arm counterpart of eval_batch — reuses _ade_per_agent /
    _flag_rate / _nanmean and the *identical* minADE/FULL_COV completion
    recipe. kl_mean = NaN (no π_θ); r_task_mean = NaN (π_ref has no GPUDrive
    task reward); collision/off-road/goal from GPUDrive Info."""
    cmask = env.cont_agent_mask
    W = cmask.shape[0]
    coll, offr, goal, rh, ent = [], [], [], [], []
    ade_runs = []
    t0 = time.time()
    for k in range(n_rollouts):
        s_live, rh_k, ent_k = ref_rollout(env, tp, dec)
        coll.append(_flag_rate(env, cmask, "collided"))
        offr.append(_flag_rate(env, cmask, "off_road"))
        goal.append(_flag_rate(env, cmask, "goal_achieved"))
        rh.append(rh_k); ent.append(ent_k)
        run = {}
        for w, sl in enumerate(s_live):
            rolled = sl["pos_xy"]; gt = gt_scenes[w]["pos_xy"]
            valid = gt_scenes[w]["valid"]; ctrl = gt_scenes[w]["cmask"]
            T = min(rolled.shape[1], gt.shape[1], valid.shape[1])
            ade, has, cov = _ade_per_agent(rolled[:, :T], gt[:, :T],
                                           valid[:, :T], ctrl)
            run[w] = (ade, has, cov)
        ade_runs.append(run)
    dt = time.time() - t0
    # minADE — identical recipe to eval_batch (FULL_COV completion gating)
    min_ades = []
    n_ctrl = n_completed = 0
    for w in range(W):
        A = ade_runs[0][w][0].shape[0]
        for a in range(A):
            if not any(ade_runs[k][w][1][a] for k in range(n_rollouts)):
                continue
            n_ctrl += 1
            full = [ade_runs[k][w][0][a] for k in range(n_rollouts)
                    if ade_runs[k][w][1][a]
                    and ade_runs[k][w][2][a] >= FULL_COV]
            if full:
                n_completed += 1
                min_ades.append(float(min(full)))
    return dict(collision_rate=_nanmean(coll),
                off_road_rate=_nanmean(offr),
                goal_rate=_nanmean(goal),
                r_task_mean=float("nan"),
                min_ade_m=float(np.mean(min_ades)) if min_ades else float("nan"),
                ade_completion_rate=(n_completed / n_ctrl) if n_ctrl
                else float("nan"),
                r_h_mean=_nanmean(rh), kl_mean=float("nan"),
                entropy_mean=_nanmean(ent),
                rollout_time_s=dt)


# ---------- entrypoint -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to trained .pt checkpoint to evaluate.")
    ap.add_argument("--scene-batches", type=int, default=12,
                    help="Number of distinct scene-batches (swap_data_batch "
                    "calls). Total scenes = scene-batches × worlds.")
    ap.add_argument("--worlds", type=int, default=8,
                    help="Parallel Madrona worlds (scenes) per batch.")
    ap.add_argument("--rollouts", type=int, default=6,
                    help="Rollouts per scene for minADE (min over these). K.")
    ap.add_argument("--ref-arm", action="store_true",
                    help="Also evaluate π_ref (CAT-K clsft_E9) closed-loop "
                    "as a third arm (Phase A.5 reference baseline).")
    ap.add_argument("--split", default="validation",
                    choices=["training", "validation", "testing"],
                    help="Dataset split to evaluate on (default: validation). "
                    "Use 'training' to test for overfitting / data-coverage.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: "
                    "/spacer/eval_runs/<ckpt-stem>/quick_metrics.json")
    a = ap.parse_args()

    n_scenes = a.scene_batches * a.worlds
    env, _ = build_env(n_scenes, n_worlds=a.worlds, split=a.split)
    print(f"[eval_quick] split: {a.split}")
    obs0 = env.reset()
    odim = obs0[env.cont_agent_mask].shape[-1]
    tp, dec = load_ref()
    print(f"[eval_quick] env: W={a.worlds} scene-batches={a.scene_batches} "
          f"rollouts/scene={a.rollouts} obs_dim={odim}")
    print(f"[eval_quick] total scenes={n_scenes} "
          f"total rollouts/arm={n_scenes * a.rollouts}")

    ttraj = tp._get_agent_shape_and_token_traj(
        torch.zeros(env.cont_agent_mask.shape[1], dtype=torch.long))[2]

    # both policies built up front; env is shared and cannot be rebuilt
    trained = TokenPolicy(obs_dim=odim).to(DEV)
    dummy_opt = torch.optim.Adam(trained.parameters(), lr=1e-3)
    it_in_ckpt, _, meta_in_ckpt = load_ckpt(a.ckpt, trained, dummy_opt)
    trained._ttraj = ttraj
    random_pol = TokenPolicy(obs_dim=odim).to(DEV)
    random_pol._ttraj = ttraj
    print(f"[eval_quick] trained ckpt: {a.ckpt} (it={it_in_ckpt})")
    if a.ref_arm:
        print("[eval_quick] ref arm ON — π_ref (clsft_E9) closed-loop")

    # per-arm per-batch accumulators
    arm_names = ["trained", "random"] + (["ref"] if a.ref_arm else [])
    agg = {arm: [] for arm in arm_names}
    batches_done = 0
    with torch.no_grad():
        for b in range(a.scene_batches):
            try:
                env.swap_data_batch()        # cycle to next W fresh scenes
            except StopIteration:
                print(f"[eval_quick] data loader exhausted after "
                      f"{b} batches — stopping early")
                break
            # GT is the WOMD-logged trajectory — extract once, used by both
            # arms and all rollouts of this batch (expert tensor is static).
            gt_scenes = [extract_gpudrive_scene(env, w)
                         for w in range(a.worlds)]
            for arm in arm_names:
                if arm == "ref":
                    agg[arm].append(
                        eval_batch_ref(env, tp, dec, gt_scenes, a.rollouts))
                else:
                    pol = trained if arm == "trained" else random_pol
                    agg[arm].append(
                        eval_batch(env, pol, tp, dec, gt_scenes, a.rollouts))
            batches_done += 1
            print(f"[eval_quick] batch {b + 1}/{a.scene_batches} done")

    # aggregate across batches
    arms = {}
    metric_keys = ["collision_rate", "off_road_rate", "goal_rate",
                   "r_task_mean", "min_ade_m", "ade_completion_rate",
                   "r_h_mean", "kl_mean", "entropy_mean"]
    for arm in arm_names:
        rows = agg[arm]
        m = {k: _nanmean([r[k] for r in rows]) for k in metric_keys}
        total_time = sum(r["rollout_time_s"] for r in rows)
        n_roll = batches_done * a.worlds * a.rollouts
        m["wall_time_s"] = total_time
        m["throughput_scenarios_per_s"] = (n_roll / total_time
                                           if total_time > 0 else float("nan"))
        m["n_scenes"] = batches_done * a.worlds
        m["n_rollouts"] = n_roll
        arms[arm] = m
    arms["trained"]["ckpt_it"] = it_in_ckpt
    arms["trained"]["ckpt_meta"] = meta_in_ckpt

    out = {
        "ckpt": a.ckpt,
        "config": {"scene_batches": batches_done, "worlds": a.worlds,
                   "rollouts_per_scene": a.rollouts,
                   "dynamics": "state", "collision_behavior": "stop",
                   "variant": "V4 (KL + r_inf)"},
        "arms": arms,
    }
    if a.out is None:
        stem = os.path.splitext(os.path.basename(a.ckpt))[0]
        out_dir = f"/spacer/eval_runs/{stem}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "quick_metrics.json")
    else:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        out_path = a.out
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval_quick] wrote {out_path}")

    # side-by-side — one column per arm, plus Δ(trained−random)
    print("\n--- summary  (↓ coll/off/r_task/minADE   ↑ goal/r_h/entropy/thru/compl) ---")
    cols = ["collision_rate", "off_road_rate", "goal_rate", "min_ade_m",
            "ade_completion_rate", "r_task_mean", "r_h_mean", "kl_mean",
            "entropy_mean", "throughput_scenarios_per_s", "wall_time_s"]
    hdr = f"{'metric':28s}" + "".join(f"  {arm:>11s}" for arm in arm_names)
    hdr += f"  {'Δ (t-r)':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for c in cols:
        line = f"{c:28s}"
        for arm in arm_names:
            v = arms[arm].get(c, float("nan"))
            line += f"  {v:>11.4f}"
        t = arms["trained"].get(c, float("nan"))
        r = arms["random"].get(c, float("nan"))
        d = (t - r) if all(isinstance(x, (int, float)) and not math.isnan(x)
                           for x in (t, r)) else float("nan")
        line += f"  {d:>+11.4f}"
        print(line)


if __name__ == "__main__":
    main()
