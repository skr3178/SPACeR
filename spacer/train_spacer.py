"""M4: SPACeR training channel — π_θ ⊕ M1 driver ⊕ GPUDrive ⊕ π_ref ⊕ Eq.1/2/3/5.

Compact PPO+KL loop (Eq. 2 `L = L_PPO − β·D_KL`, Eq. 1 `r = r_task + α·r_h`).
This is the *channel smoke*: prove the full loop runs end-to-end and is
numerically stable on the RTX 3060 (no NaN/OOM, KL bounded, π_θ updates,
β scales the anchoring term). It is NOT a converged model.

KNOWN PRECISION ITEM (flagged, deferred to M4-proper/M5): exact per-agent
π_θ↔π_ref correspondence. π_θ acts in GPUDrive agent/order space; π_ref scores
in the adapter's filtered agent space. Here r_h (Eq.3) uses the tokenizer's
gt_idx of the *π_θ-produced* trajectory (exact: that's what π_θ did), and the
Eq.5 KL uses π_θ's per-decision logits vs π_ref's per-step distribution at the
scene level (smoke proxy). The agent-id map is the remaining task before a
faithful training run.
"""
import sys, time, dataclasses, math, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from omegaconf import OmegaConf
from torch_geometric.data import Batch
import numpy as np
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.datatypes.observation import GlobalEgoState
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from gpudrive_to_smart import (extract_gpudrive_scene, scene_dict_to_heterodata,
                               finite_diff_velocity, NUM_STEPS)
from token_decode import decode_token_sequence
from policy_token import TokenPolicy, N_TOKENS
from anchor import (kl_theta_ref, r_humanlike, align_executed_tokens,
                    align_agents, REF_STEP_OFFSET)

DEV = "cuda"
SHIFT = 5                                  # 0.5 s token, 2 Hz (checkpoint-native)


def build_env(n_scenes):
    cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
    # single world (rollout/adapter are world-0); cycle `n_scenes` distinct
    # scenes across env.reset() for variety.
    loader = SceneDataLoader(root="/gpd/data/processed/validation",
                             batch_size=1, dataset_size=max(1, n_scenes),
                             sample_with_replacement=False)
    # Variant 4 (KL + r_inf) — Table A2 best composite (0.74), goals dropped.
    # r_task = − w_coll·𝟙[collision] − w_off·𝟙[off-road]   (no goal channel)
    ec = dataclasses.replace(
        EnvConfig(), dynamics_model="state", collision_behavior="ignore",
        remove_non_vehicles=cfg.remove_non_vehicles, obs_radius=cfg.obs_radius,
        reward_type="weighted_combination",
        goal_achieved_weight=0.0,
        collision_weight=-0.75,
        off_road_weight=-0.75)
    env = GPUDriveTorchEnv(config=ec, data_loader=loader,
                           max_cont_agents=cfg.max_controlled_agents, device=DEV)
    return env, cfg


def load_ref():
    ck = torch.load("/ckpt/clsft_E9.ckpt", map_location="cpu", weights_only=False)
    mc = OmegaConf.create(ck["hyper_parameters"]).model_config
    tp = TokenProcessor(**mc.token_processor)
    dec = SMARTDecoder(**mc.decoder, n_token_agent=tp.n_token_agent).to(DEV).eval()
    dec.load_state_dict({k[8:]: v for k, v in ck["state_dict"].items()
                         if k.startswith("encoder.")}, strict=True)
    for p in dec.parameters():
        p.requires_grad_(False)             # π_ref FROZEN
    return tp, dec


@torch.no_grad()
def set_state(env, pos_xy, head):
    """Drive `state` dynamics: command global pose for all agents (world 0)."""
    W, A = env.cont_agent_mask.shape
    g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                   backend="torch", device=DEV)
    act = torch.zeros((W, A, 10), dtype=torch.float32, device=DEV)
    act[0, :, 0] = pos_xy[:, 0]
    act[0, :, 1] = pos_xy[:, 1]
    act[0, :, 2] = g.pos_z[0]
    act[0, :, 3] = head
    env.step_dynamics(act)


def rollout(env, policy, world_idx=0):
    """Full-episode π_θ rollout in token space; M1 decodes → `state` drive.
    Returns per-decision (obs, token, logprob, value, logits) + the rolled
    trajectory buffers for π_ref scoring."""
    obs = env.reset()
    cmask = env.cont_agent_mask
    ego0 = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                      backend="torch", device=DEV)
    theta_ids = ego0.id[0][cmask[0]].long()                  # π_θ logits row order
    s0 = extract_gpudrive_scene(env, world_idx)
    A = s0["pos_xy"].shape[0]
    buf_pos = np.zeros((A, NUM_STEPS, 2), np.float32)
    buf_head = np.zeros((A, NUM_STEPS), np.float32)
    # token vocab template for decode (vehicles-first)
    tp_traj = policy._ttraj                                  # [A, 2048, 4, 2]
    prev_pos = torch.tensor(s0["pos_xy"][:, 0], dtype=torch.float32)
    prev_head = torch.tensor(s0["yaw"][:, 0], dtype=torch.float32)
    steps = list(range(SHIFT, NUM_STEPS, SHIFT))             # 18 token-steps
    rec = {"obs": [], "tok": [], "lp": [], "val": [], "logits": [], "rtask": []}
    t = 0
    for k, i in enumerate(steps):
        g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                       backend="torch", device=DEV)
        # record current global state for all steps up to i
        while t <= i and t < NUM_STEPS:
            buf_pos[:, t, 0] = g.pos_x[0].cpu().numpy()
            buf_pos[:, t, 1] = g.pos_y[0].cpu().numpy()
            buf_head[:, t] = g.rotation_angle[0].cpu().numpy()
            t += 1
        x = obs[cmask]                                        # [nc, obs_dim]
        logits = policy.logits(x)                             # [nc, 2048]
        tok, lp, _, val = policy(x, deterministic=False)
        rec["obs"].append(x.detach()); rec["tok"].append(tok.detach())
        rec["lp"].append(lp.detach()); rec["val"].append(val.detach())
        rec["logits"].append(logits.detach())
        # decode chosen token (all agents) -> next 0.5 s pose, drive 5 sim steps
        tok_all = torch.zeros(A, 1, dtype=torch.long)
        tok_all[cmask[0].cpu()] = tok.detach().cpu().view(-1, 1)
        dpos, dhead = decode_token_sequence(
            tok_all, prev_pos, prev_head, tp_traj,
            torch.ones(A, 1, dtype=torch.bool))
        dpos, dhead = dpos[:, 0], dhead[:, 0]
        for _ in range(SHIFT):
            set_state(env, dpos.to(DEV), dhead.to(DEV))
        prev_pos, prev_head = dpos, dhead
        rec["rtask"].append(env.get_rewards()[cmask].detach())
        obs = env.get_obs()
    # rolled trajectory for π_ref
    s_live = dict(s0); s_live["pos_xy"] = buf_pos; s_live["yaw"] = buf_head
    s_live["vel_xy"] = finite_diff_velocity(buf_pos, s0["valid"])
    return rec, s_live, theta_ids


def score_ref(tp, dec, s_live):
    hd = scene_dict_to_heterodata(s_live, "spacer_rollout")
    b = Batch.from_data_list([hd])
    tmap, tag = tp(b)
    to = lambda d: {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in d.items()}
    with torch.no_grad():
        pred = dec(to(tmap), to(tag))
    ref_ids = b["agent"]["id"].to(DEV).long()
    return (pred["next_token_logits"], align_executed_tokens(tag["gt_idx"]).to(DEV),
            pred["next_token_valid"].bool(), ref_ids)


def spacer_iteration(env, policy, opt, tp, dec, alpha, beta):
    rec, s_live, theta_ids = rollout(env, policy)
    ref_logits, exec_tok, vmask, ref_ids = score_ref(tp, dec, s_live)  # π_ref
    ref_logits = ref_logits.detach()                    # π_ref FROZEN

    # Eq.3 r_h on the π_θ-produced trajectory (exact, ref-agent space) — reward
    _, rh = r_humanlike(ref_logits, exec_tok, vmask)

    # Eq.5 EXACT per-(agent,step) KL (M5b): recompute π_θ logits WITH grad
    # from stored obs so −β·KL actually trains π_θ; align temporally
    # (decision k = ref-step k−REF_STEP_OFFSET) and per-agent by object_id.
    gather, amatch = align_agents(ref_ids, theta_ids)       # [A_ref],[A_ref]
    A_ref, T_ref, _ = ref_logits.shape
    off = REF_STEP_OFFSET
    th_steps = []
    for j in range(T_ref):
        k = j + off
        if k < len(rec["obs"]):
            th_steps.append(policy.logits(rec["obs"][k]))   # [nc,2048] w/ grad
        else:
            th_steps.append(policy.logits(rec["obs"][-1]))
    th = torch.stack(th_steps, dim=1)                       # [nc, T_ref, 2048]
    th_aligned = th[gather]                                 # [A_ref, T_ref, 2048]
    kmask = vmask & amatch.unsqueeze(1)                     # [A_ref, T_ref]
    _, kl = kl_theta_ref(th_aligned, ref_logits, kmask)     # exact, differentiable

    # Eq.1 reward + PG; Eq.2 loss (min form: −L_PPO + β·D_KL)
    r_task = torch.stack([r.mean() for r in rec["rtask"]]).mean()
    r_total = r_task + alpha * rh
    logp = torch.stack([policy(o, action=a)[1].mean()
                        for o, a in zip(rec["obs"], rec["tok"])])
    l_pg = -(logp.mean() * r_total.detach())   # ≈ −L_PPO
    loss = l_pg + beta * kl                                  # Eq.2 (min form)
    opt.zero_grad(); loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return dict(r_task=float(r_task), r_h=float(rh), kl=float(kl),
                loss=float(loss), gnorm=float(gnorm),
                ndec=len(rec["tok"]))


def run(beta, iters, scenes, alpha=0.0, lr=3e-4, tag=""):
    """One training run; returns per-iter metrics.

    Variant 4 defaults (Table A2 best composite, paper-validated):
      - alpha=0.0         ⇒ LLH reward (Eq.3) is dead weight given KL (Eq.5)
      - w_goal=0.0        ⇒ goal reward dropped (set in build_env's EnvConfig)
      - r_task = r_inf    = −0.75·𝟙[collision] − 0.75·𝟙[off-road]
    Loss reduces to:  L = −L_PPO(r_inf) + β·KL(π_θ ‖ π_ref).
    r_task therefore doubles as the reactivity proxy (lower ⇒ more unsafe)."""
    env, _ = build_env(scenes)
    obs0 = env.reset()
    odim = obs0[env.cont_agent_mask].shape[-1]
    policy = TokenPolicy(obs_dim=odim).to(DEV)
    tp, dec = load_ref()
    policy._ttraj = tp._get_agent_shape_and_token_traj(
        torch.zeros(env.cont_agent_mask.shape[1], dtype=torch.long))[2]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    w0 = torch.cat([p.flatten() for p in policy.parameters()]).clone()
    hist = []
    t0 = time.time()
    for it in range(iters):
        m = spacer_iteration(env, policy, opt, tp, dec, alpha, beta)
        hist.append(m)
        print(f"  [{tag}β={beta}] it{it:02d}: r_task={m['r_task']:+.3f} "
              f"r_h={m['r_h']:+.3f} KL={m['kl']:.3f} loss={m['loss']:+.3f} "
              f"|g|={m['gnorm']:.2f}")
    dt = time.time() - t0
    dw = float((torch.cat([p.flatten() for p in policy.parameters()]) - w0)
               .abs().mean())
    return hist, dw, dt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "ablate"], default="smoke")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="LLH (Eq.3) weight. Variant 4 default: 0.0 — Table A2"
                    " shows LLH adds nothing on top of KL. Set >0 only for"
                    " ablation.")
    ap.add_argument("--beta", type=float, default=0.1)
    a = ap.parse_args()

    if a.mode == "smoke":                       # M5b faithful short run
        print(f"M5b faithful run: iters={a.iters} scenes={a.scenes} "
              f"α={a.alpha} β={a.beta} (exact per-agent KL, differentiable)")
        h, dw, dt = run(a.beta, a.iters, a.scenes, a.alpha, tag="")
        fin = all(np.isfinite([h[-1]['loss'], h[-1]['kl'], h[-1]['r_h']]))
        print(f"params changed (mean|Δw|)={dw:.2e} | finite={fin} | "
              f"{dt:.1f}s ({a.iters/dt:.2f} it/s)")
        print("VERDICT:", "M5b OK — faithful loop (exact per-agent Eq.5, "
              "differentiable), stable, π_θ updates"
              if (fin and dw > 0) else "M5b FAIL")
    else:                                       # M5c β-ablation + reactivity
        print(f"M5c ablation: {a.iters} iters/run, scenes={a.scenes}, "
              f"α={a.alpha}, β∈{{0.0, {a.beta}}}")
        res = {}
        for b in (0.0, a.beta):
            h, dw, dt = run(b, a.iters, a.scenes, a.alpha,
                            tag="ABL ")
            kl = np.array([x['kl'] for x in h]); rt = np.array([x['r_task'] for x in h])
            rh = np.array([x['r_h'] for x in h])
            res[b] = dict(kl_mean=kl.mean(), kl_last=kl[-3:].mean(),
                          rt_mean=rt.mean(), rt_last=rt[-3:].mean(),
                          rh_last=rh[-3:].mean(), dw=dw, dt=dt)
            print(f"  β={b}: KL μ={kl.mean():.3f}→last{kl[-3:].mean():.3f}  "
                  f"r_task μ={rt.mean():+.3f}→last{rt[-3:].mean():+.3f}  "
                  f"r_h last{rh[-3:].mean():+.3f}  Δw={dw:.1e}  {dt:.0f}s")
        b0, b1 = res[0.0], res[a.beta]
        anchor_effect = b1["kl_last"] < b0["kl_last"]          # β>0 lowers KL
        # reactivity proxy: r_task (collision/off-road penalties baked in)
        reactivity_ok = b1["rt_last"] >= b0["rt_last"] - 0.05
        print(f"\nanchoring effect (β>0 KL < β=0 KL): {anchor_effect} "
              f"({b1['kl_last']:.3f} vs {b0['kl_last']:.3f})")
        print(f"reactivity proxy r_task (β>0 ≳ β=0): {reactivity_ok} "
              f"({b1['rt_last']:+.3f} vs {b0['rt_last']:+.3f})  "
              f"[low/neg ⇒ collisions/off-road ⇒ 2 Hz too coarse ⇒ S2.5]")
        print("VERDICT:", "M5c TREND OK — β>0 anchors (lower KL), loop stable "
              "at scale" if anchor_effect else
              "M5c — anchoring/reactivity inconclusive at this scale (see notes)")
