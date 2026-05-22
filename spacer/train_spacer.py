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
import os
# M5d: avoid VRAM fragmentation in the W=64 π_ref batched forward (12 GB 3060).
# Must be set before torch is imported. PyTorch's recommended fix for the same
# OOM message we observed at W=64 without it (~647 MB reserved-but-unallocated).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys, time, dataclasses, math, random, torch
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


def build_env(n_scenes, n_worlds=1, split="validation", data_root=None):
    cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
    # `n_worlds` parallel Madrona worlds; `n_scenes` ≥ `n_worlds` distinct scenes
    # cycled across env.reset() (paper runs 64+ worlds; here we let the caller
    # pick, default 1 for back-compat with the M5a–c single-world path).
    # `split` selects the GPUDrive_mini folder: "training" (1000 scenes) for
    # training, "validation" (150, held out) for eval. Default "validation"
    # keeps pre-split callers/tests unchanged.
    # `data_root`: explicit scene directory (overrides `split`). Used for the
    # new 10k dataset — pass e.g. /data_new/training/group_0 (the dir holding
    # the tfrecord-*.json files directly). None ⇒ legacy GPUDrive_mini path.
    root = data_root if data_root else f"/gpd/data/processed/{split}"
    loader = SceneDataLoader(root=root,
                             batch_size=n_worlds,
                             dataset_size=max(n_worlds, n_scenes),
                             sample_with_replacement=False)
    # Variant 4 (KL + r_inf) — Table A2 best composite (0.74), goals dropped.
    # r_task = − w_coll·𝟙[collision] − w_off·𝟙[off-road]   (no goal channel)
    # collision_behavior="stop": LEVEL-triggered penalty (sustained while in
    # collided/off-road state). "ignore" makes the same flags EDGE-triggered
    # (fire once on entry, clear next step) — verified by
    # test_rtask_diagnostic.py; "stop" gives a much stronger RL gradient.
    ec = dataclasses.replace(
        EnvConfig(), dynamics_model="state", collision_behavior="stop",
        remove_non_vehicles=cfg.remove_non_vehicles, obs_radius=cfg.obs_radius,
        reward_type="weighted_combination",
        goal_achieved_weight=0.0,
        collision_weight=-0.75,
        off_road_weight=-0.75)
    env = GPUDriveTorchEnv(config=ec, data_loader=loader,
                           max_cont_agents=cfg.max_controlled_agents, device=DEV)
    return env, cfg


def _scene_pool(root, size):
    """The fixed, deterministic scene pool: first `size` tfrecord files in
    `root`, sorted — matches SceneDataLoader's own selection so the injection
    pool is consistent with the env's initial batch."""
    files = sorted(f for f in os.listdir(root) if f.startswith("tfrecord"))
    return [os.path.join(root, f) for f in files[:size]]


def inject_scenes(env, pool, n_inject, rng):
    """Paper-style partial scene injection. Drops the oldest `n_inject` of the
    W worlds, draws `n_inject` fresh scenes from `pool`, and swaps the W-world
    batch in place (FIFO sliding window). Follows GPUDrive's resample pattern
    (env_puffer.resample_scenario_batch): swap_data_batch reinitialises the sim
    maps + cont_agent_mask; the caller's next env.reset() resets sim state."""
    cur = list(env.data_batch)                       # current W scene paths
    W = len(cur)
    n = max(0, min(n_inject, W))
    if n == 0:
        return
    fresh = [rng.choice(pool) for _ in range(n)]
    env.swap_data_batch(cur[n:] + fresh)             # retained (W−n) + fresh


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
def set_state(env, pos_per_w, head_per_w):
    """Drive `state` dynamics: command global pose for all worlds at once.

    pos_per_w  : list[Tensor[A_w, 2]] of length W
    head_per_w : list[Tensor[A_w]] of length W
    A single env.step_dynamics(act) advances all W worlds simultaneously.
    """
    W, A = env.cont_agent_mask.shape
    g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                   backend="torch", device=DEV)
    act = torch.zeros((W, A, 10), dtype=torch.float32, device=DEV)
    for w in range(W):
        Aw = pos_per_w[w].shape[0]
        act[w, :Aw, 0] = pos_per_w[w][:, 0].to(DEV)
        act[w, :Aw, 1] = pos_per_w[w][:, 1].to(DEV)
        act[w, :Aw, 2] = g.pos_z[w, :Aw]
        act[w, :Aw, 3] = head_per_w[w].to(DEV)
    env.step_dynamics(act)


def rollout(env, policy):
    """Full-episode π_θ rollout in token space across all W worlds in parallel.

    Policy forward stays flat over [nc_total, obs_dim] (Madrona ego/partner/
    road obs are agent-decentralized; the policy is agent-count-agnostic).
    Per-world side (scene extract, decode, rolled buffers, set_state) is split
    into a list of W items because each world has its own scene + agent count.

    Returns per-decision (obs, token, logprob, value, logits) flat across
    worlds, plus the rolled trajectories / theta_ids / nc_per_w needed to score
    π_ref per world."""
    obs = env.reset()
    cmask = env.cont_agent_mask                              # [W, A_max]
    W, A_max = cmask.shape
    ego0 = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                      backend="torch", device=DEV)
    # per-world π_θ logits row order (controlled agents only, in cmask order)
    theta_ids_per_w = [ego0.id[w][cmask[w]].long() for w in range(W)]
    nc_per_w = [int(cmask[w].sum().item()) for w in range(W)]
    # per-world initial scenes (the adapter is already per-world via world_idx)
    scenes0 = [extract_gpudrive_scene(env, w) for w in range(W)]
    A_per_w = [s["pos_xy"].shape[0] for s in scenes0]
    buf_pos = [np.zeros((A_per_w[w], NUM_STEPS, 2), np.float32) for w in range(W)]
    buf_head = [np.zeros((A_per_w[w], NUM_STEPS), np.float32) for w in range(W)]
    # vehicles-first token templates (all type 0); A_max matches each world's
    # padded agent count, so one [A_max, 2048, 4, 2] template slices for any W
    tp_traj = policy._ttraj
    prev_pos = [torch.tensor(scenes0[w]["pos_xy"][:, 0], dtype=torch.float32)
                for w in range(W)]
    prev_head = [torch.tensor(scenes0[w]["yaw"][:, 0], dtype=torch.float32)
                 for w in range(W)]
    steps = list(range(SHIFT, NUM_STEPS, SHIFT))             # 18 token-steps
    rec = {"obs": [], "tok": [], "lp": [], "val": [], "logits": [], "rtask": []}
    t = 0
    for k, i in enumerate(steps):
        g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                       backend="torch", device=DEV)
        # record current global state per world for all steps up to i
        while t <= i and t < NUM_STEPS:
            for w in range(W):
                Aw = A_per_w[w]
                buf_pos[w][:, t, 0] = g.pos_x[w, :Aw].cpu().numpy()
                buf_pos[w][:, t, 1] = g.pos_y[w, :Aw].cpu().numpy()
                buf_head[w][:, t] = g.rotation_angle[w, :Aw].cpu().numpy()
            t += 1
        x = obs[cmask]                                        # [nc_total, obs_dim]
        logits = policy.logits(x)                             # [nc_total, 2048]
        tok, lp, _, val = policy(x, deterministic=False)
        rec["obs"].append(x.detach()); rec["tok"].append(tok.detach())
        rec["lp"].append(lp.detach()); rec["val"].append(val.detach())
        rec["logits"].append(logits.detach())
        # split chosen tokens per world, decode each to next 0.5 s pose
        tok_split = list(torch.split(tok.detach().cpu(), nc_per_w))
        dpos_per_w, dhead_per_w = [], []
        for w in range(W):
            Aw = A_per_w[w]
            tok_w = torch.zeros(Aw, 1, dtype=torch.long)
            tok_w[cmask[w].cpu()[:Aw]] = tok_split[w].view(-1, 1)
            dp, dh = decode_token_sequence(
                tok_w, prev_pos[w], prev_head[w], tp_traj[:Aw],
                torch.ones(Aw, 1, dtype=torch.bool))
            dpos_per_w.append(dp[:, 0]); dhead_per_w.append(dh[:, 0])
        for _ in range(SHIFT):
            set_state(env, dpos_per_w, dhead_per_w)
        prev_pos, prev_head = dpos_per_w, dhead_per_w
        # NOTE: env.get_rewards() takes weights as ARGS (defaults −0.5/+1.0/
        # −0.5) — it does NOT read EnvConfig. Passing the EnvConfig weights
        # explicitly is what actually applies Variant 4 (w_goal=0, ±0.75);
        # calling it bare silently trained on goal-reward-ON. (Test 19.)
        rec["rtask"].append(env.get_rewards(
            collision_weight=env.config.collision_weight,
            goal_achieved_weight=env.config.goal_achieved_weight,
            off_road_weight=env.config.off_road_weight)[cmask].detach())
        obs = env.get_obs()
    # rolled trajectories for π_ref (one s_live dict per world)
    s_live_per_w = []
    for w in range(W):
        sl = dict(scenes0[w])
        sl["pos_xy"] = buf_pos[w]; sl["yaw"] = buf_head[w]
        sl["vel_xy"] = finite_diff_velocity(buf_pos[w], scenes0[w]["valid"])
        s_live_per_w.append(sl)
    return rec, s_live_per_w, theta_ids_per_w, nc_per_w


def score_ref(tp, dec, s_live_per_w):
    """Batched π_ref forward across W worlds. Single dec(...) call processes
    all W scenes; b["agent"]["batch"] is then used to split outputs per world.

    Returns list[(ref_logits, exec_tok, vmask, ref_ids)] of length W.
    """
    hds = [scene_dict_to_heterodata(s, f"spacer_rollout_{w}")
           for w, s in enumerate(s_live_per_w)]
    b = Batch.from_data_list(hds)
    tmap, tag = tp(b)
    to = lambda d: {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in d.items()}
    with torch.no_grad():
        pred = dec(to(tmap), to(tag))
    agent_batch = b["agent"]["batch"].to(DEV).long()         # [A_ref_total]
    ref_ids_all = b["agent"]["id"].to(DEV).long()
    exec_tok_all = align_executed_tokens(tag["gt_idx"]).to(DEV)
    logits_all = pred["next_token_logits"]
    vmask_all = pred["next_token_valid"].bool()
    out = []
    for w in range(len(s_live_per_w)):
        m = agent_batch == w
        out.append((logits_all[m], exec_tok_all[m], vmask_all[m], ref_ids_all[m]))
    return out


# ---------------------------------------------------------------------------
# PPO update — ported from GPUDrive's PufferLib PPO
# (`gpudrive/integrations/puffer/ppo.py`), the optimiser the SPACeR paper uses.
# Algorithmic hyperparameters are paper Table A3 (verbatim). Scale params
# (total steps, batch, minibatch) are emergent from --iters / --worlds — see
# Training_Config.md. The closed-form KL anchor (Eq. 5) is added to the PPO
# loss (Eq. 2); Variant 4 uses reward = r_task (α=0), KL as a loss term.
# ---------------------------------------------------------------------------
PPO_GAMMA       = 0.99      # discount factor
PPO_GAE_LAMBDA  = 0.95      # GAE λ
PPO_CLIP_COEF   = 0.2       # policy ratio clip
PPO_VF_COEF     = 0.3       # value-loss weight
PPO_ENT_COEF    = 1e-4      # entropy bonus
PPO_MAX_GRAD    = 0.5       # grad-norm clip (Table A3; was 1.0 in compact loop)
PPO_EPOCHS      = 4         # optimisation epochs per rollout
PPO_N_MINIBATCH = 16        # minibatches per epoch (paper Table A3: 131072/8192)
PPO_NORM_ADV    = True      # normalise advantages


def _gae(rew, val, gamma, lam):
    """GAE-λ per agent along the token-decision axis.
    rew, val : [T, N] (one episode; value bootstrap = 0 past the end).
    Returns (advantages[T,N], returns[T,N])."""
    T, N = rew.shape
    adv = torch.zeros_like(rew)
    last = torch.zeros(N, device=rew.device)
    zero = torch.zeros(N, device=rew.device)
    for t in range(T - 1, -1, -1):
        nextv = val[t + 1] if t + 1 < T else zero
        delta = rew[t] + gamma * nextv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def spacer_iteration(env, policy, opt, tp, dec, alpha, beta):
    """One SPACeR training iteration: self-play rollout → π_ref scoring →
    **PPO update** (GPUDrive PufferLib PPO, Table A3) with the closed-form
    KL anchor (Eq. 5) added to the loss (Eq. 2)."""
    rec, s_live_per_w, theta_ids_per_w, nc_per_w = rollout(env, policy)
    per_world = score_ref(tp, dec, s_live_per_w)
    W = len(per_world)
    T = len(rec["tok"])
    N = rec["tok"][0].shape[0]                       # nc_total (flat over W)
    off = REF_STEP_OFFSET

    # --- stack rollout records → [T, N] (all detached; PPO recomputes) -----
    obs   = torch.stack(rec["obs"])                  # [T, N, D]
    act   = torch.stack(rec["tok"])                  # [T, N]
    oldlp = torch.stack(rec["lp"])                   # [T, N]
    val   = torch.stack(rec["val"]).reshape(T, N)    # [T, N]
    rew   = torch.stack([r.reshape(N) for r in rec["rtask"]])      # [T, N]
    roll_logits = torch.stack(rec["logits"])         # [T, N, 2048] rollout-time

    # --- GAE (Variant 4: GAE reward = r_task) ------------------------------
    adv, ret = _gae(rew, val, PPO_GAMMA, PPO_GAE_LAMBDA)

    # --- π_ref logits scattered into the flat [T, N] order -----------------
    # ref-step j ↔ decision k = j + REF_STEP_OFFSET; per-agent align by id.
    ref_logits = torch.zeros(T, N, policy.n_tokens, device=DEV)
    ref_valid  = torch.zeros(T, N, dtype=torch.bool, device=DEV)
    nc_off = [0]
    for n in nc_per_w:
        nc_off.append(nc_off[-1] + n)
    rh_list = []
    for w, (rl, exec_tok, vmask, ref_ids) in enumerate(per_world):
        rl = rl.detach()                             # π_ref FROZEN
        if rl.numel() == 0 or nc_per_w[w] == 0:
            continue
        _, rh_w = r_humanlike(rl, exec_tok, vmask)   # Eq.3 (logged)
        rh_list.append(rh_w)
        gather, amatch = align_agents(ref_ids, theta_ids_per_w[w])
        if not amatch.any():
            continue
        A_ref, T_ref, _ = rl.shape
        Te = min(T_ref, T - off)
        if Te <= 0:
            continue
        mi = amatch.nonzero(as_tuple=True)[0]        # matched ref agents
        cols = nc_off[w] + gather[mi].long()         # their π_θ flat columns
        ref_logits[off:off + Te, cols] = rl[mi, :Te].transpose(0, 1)
        ref_valid[off:off + Te, cols] = vmask[mi, :Te].transpose(0, 1)
    rh = torch.stack(rh_list).mean() if rh_list else torch.zeros((), device=DEV)

    # --- rollout-time KL / entropy (logged — comparable to Tests 12-16) ----
    with torch.no_grad():
        lp_th = torch.log_softmax(roll_logits.float(), dim=-1)
        lp_rf = torch.log_softmax(ref_logits.float(), dim=-1)
        kl_pa  = (lp_th.exp() * (lp_th - lp_rf)).sum(-1)        # [T, N]
        ent_pa = -(lp_th.exp() * lp_th).sum(-1)                 # [T, N]
        kl_log  = (kl_pa[ref_valid].mean() if ref_valid.any()
                   else torch.zeros((), device=DEV))
        ent_log = (ent_pa[ref_valid].mean() if ref_valid.any()
                   else ent_pa.mean())
    r_task = rew.mean()

    # --- PPO update: update_epochs × minibatches (Table A3) ----------------
    flat = lambda x: x.reshape(T * N, *x.shape[2:])
    f_obs, f_act   = flat(obs), flat(act)
    f_oldlp        = flat(oldlp)
    f_adv, f_ret   = flat(adv), flat(ret)
    f_reflog, f_rv = flat(ref_logits), flat(ref_valid)
    n_samp = T * N
    mb = max(1, n_samp // PPO_N_MINIBATCH)
    loss_acc = pg_acc = v_acc = klu_acc = 0.0
    gnorm = 0.0
    nupd = 0
    for _ep in range(PPO_EPOCHS):
        perm = torch.randperm(n_samp, device=DEV)
        for s0 in range(0, n_samp, mb):
            idx = perm[s0:s0 + mb]
            o, a = f_obs[idx], f_act[idx]
            _, newlp, entropy, newval = policy(o, action=a)     # PPO forward
            newval = newval.reshape(-1)
            ratio = (newlp - f_oldlp[idx]).exp()
            A = f_adv[idx]
            if PPO_NORM_ADV:
                A = (A - A.mean()) / (A.std() + 1e-8)
            pg = torch.max(-A * ratio,
                           -A * torch.clamp(ratio, 1 - PPO_CLIP_COEF,
                                            1 + PPO_CLIP_COEF)).mean()
            v_loss = 0.5 * ((newval - f_ret[idx]) ** 2).mean()
            # closed-form KL anchor (Eq. 5) on this minibatch — π_θ moves each
            # update, π_ref is frozen, so recompute π_θ logits here.
            rv = f_rv[idx]
            if rv.any():
                lpth = torch.log_softmax(policy.logits(o), dim=-1)
                lprf = torch.log_softmax(f_reflog[idx], dim=-1)
                kl_mb = (lpth.exp() * (lpth - lprf)).sum(-1)[rv].mean()
            else:
                kl_mb = torch.zeros((), device=DEV)
            # Eq.2 (min form): −L_PPO + β·KL ; −L_PPO = pg − ent·c_ent + v·c_vf
            loss = (pg - PPO_ENT_COEF * entropy.mean()
                    + PPO_VF_COEF * v_loss + beta * kl_mb)
            opt.zero_grad(); loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                policy.parameters(), PPO_MAX_GRAD))
            opt.step()
            loss_acc += float(loss); pg_acc += float(pg)
            v_acc += float(v_loss); klu_acc += float(kl_mb); nupd += 1

    return dict(r_task=float(r_task), r_h=float(rh), kl=float(kl_log),
                ent=float(ent_log), loss=loss_acc / nupd, gnorm=gnorm,
                pg=pg_acc / nupd, vloss=v_acc / nupd, kl_upd=klu_acc / nupd,
                ndec=T, worlds=W)


def save_ckpt(path, policy, opt, it, hist, meta):
    """Atomic-ish: write to a `.tmp` then rename, so a kill mid-write doesn't
    leave a half-file that confuses `--resume`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(dict(policy=policy.state_dict(), opt=opt.state_dict(),
                    it=it, hist=hist, meta=meta), tmp)
    os.replace(tmp, path)


def load_ckpt(path, policy, opt):
    """Restore policy + optimizer; return (next_it, hist, meta)."""
    d = torch.load(path, map_location=DEV, weights_only=False)
    policy.load_state_dict(d["policy"])
    opt.load_state_dict(d["opt"])
    return int(d["it"]), list(d.get("hist", [])), dict(d.get("meta", {}))


def run(beta, iters, scenes, alpha=0.0, lr=3e-4, tag="", n_worlds=1,
        ckpt_dir=None, ckpt_every=0, resume=None, split="training",
        data_root=None, inject_n=0, inject_every=1):
    """One training run; returns per-iter metrics.

    Variant 4 defaults (Table A2 best composite, paper-validated):
      - alpha=0.0         ⇒ LLH reward (Eq.3) is dead weight given KL (Eq.5)
      - w_goal=0.0        ⇒ goal reward dropped (set in build_env's EnvConfig)
      - r_task = r_inf    = −0.75·𝟙[collision] − 0.75·𝟙[off-road]
    Loss reduces to:  L = −L_PPO(r_inf) + β·KL(π_θ ‖ π_ref).
    r_task therefore doubles as the reactivity proxy (lower ⇒ more unsafe).

    n_worlds   : number of Madrona worlds simulated in parallel per iter (M5d).
                 Default 1 ⇒ identical to the pre-M5d single-world code path.
    ckpt_dir   : if set, write checkpoints here every `ckpt_every` iters; also
                 always writes a `latest.pt` after each save.
    ckpt_every : save period in iters (0 disables; default 0).
    resume     : path to a checkpoint to load before training starts. The
                 (β, α, n_worlds) in the ckpt's meta are sanity-checked
                 against the current call; mismatch warns but does not abort.
    data_root  : explicit scene directory (overrides `split`) — for the 10k
                 dataset, e.g. /data_new/training/group_0.
    inject_n   : paper-style scene injection — # of the W worlds refreshed with
                 fresh scenes each cycle. 0 ⇒ disabled (legacy fixed-batch).
    inject_every : inject every N iterations (1 = every iter, paper's
                 "every batch").
    """
    torch.manual_seed(42)                       # Table A3: seed = 42
    env, _ = build_env(scenes, n_worlds=n_worlds, split=split,
                       data_root=data_root)
    obs0 = env.reset()
    odim = obs0[env.cont_agent_mask].shape[-1]
    policy = TokenPolicy(obs_dim=odim).to(DEV)
    tp, dec = load_ref()
    # one shared token template; vehicles-first, padded to A_max. Per-world
    # decode slices [:A_w] from this (all worlds share A_max from Madrona).
    policy._ttraj = tp._get_agent_shape_and_token_traj(
        torch.zeros(env.cont_agent_mask.shape[1], dtype=torch.long))[2]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    meta = dict(beta=beta, alpha=alpha, n_worlds=n_worlds, lr=lr)
    start_it, hist = 0, []
    if resume:
        start_it, hist, m_old = load_ckpt(resume, policy, opt)
        for k in ("beta", "alpha", "n_worlds"):
            if m_old.get(k) != meta[k]:
                print(f"  [resume] WARNING meta mismatch on {k}: "
                      f"ckpt={m_old.get(k)} now={meta[k]}")
        print(f"  [resume] loaded {resume} → resuming at it{start_it}, "
              f"{len(hist)} prior metrics")
    w0 = torch.cat([p.flatten() for p in policy.parameters()]).clone()
    # paper-style partial scene injection (FIFO sliding window over a fixed pool)
    pool, inj_rng = None, None
    if inject_n > 0:
        root = data_root if data_root else f"/gpd/data/processed/{split}"
        pool = _scene_pool(root, scenes)
        inj_rng = random.Random(42)
        print(f"  [inject] scene injection ON: {inject_n}/{n_worlds} worlds "
              f"refreshed every {inject_every} iter(s); pool={len(pool)} "
              f"scenes ({root})")
    t0 = time.time()
    for it in range(start_it, iters):
        if (inject_n > 0 and it > start_it
                and (it - start_it) % inject_every == 0):
            inject_scenes(env, pool, inject_n, inj_rng)
        m = spacer_iteration(env, policy, opt, tp, dec, alpha, beta)
        hist.append(m)
        print(f"  [{tag}β={beta} W={n_worlds}] it{it:02d}: "
              f"r_task={m['r_task']:+.3f} r_h={m['r_h']:+.3f} "
              f"KL={m['kl']:.3f} H={m['ent']:.3f} "
              f"pg={m['pg']:+.3f} vL={m['vloss']:.3f} "
              f"loss={m['loss']:+.3f} |g|={m['gnorm']:.2f}")
        if ckpt_dir and ckpt_every > 0 and ((it + 1) % ckpt_every == 0
                                             or (it + 1) == iters):
            fn = os.path.join(ckpt_dir,
                              f"ckpt_b{beta}_W{n_worlds}_it{it+1:06d}.pt")
            save_ckpt(fn, policy, opt, it + 1, hist, meta)
            save_ckpt(os.path.join(ckpt_dir, "latest.pt"),
                      policy, opt, it + 1, hist, meta)
            print(f"  [ckpt] saved {fn}")
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
    ap.add_argument("--worlds", type=int, default=48,
                    help="Parallel Madrona worlds per iter (M5d). Default 48 "
                    "— Test 17 ceiling sweep: W=48 is the safe max on the "
                    "12 GB 3060 (~10.5 GB peak, ~1.8 GB headroom); W=64 is the "
                    "hard ceiling (~11.9 GB, ~0.4 GB free — risky for long "
                    "runs); W=80 OOMs. Pass --worlds 1 to reproduce the "
                    "pre-M5d single-world path.")
    ap.add_argument("--ckpt-dir", default="/spacer/checkpoints",
                    help="Directory for checkpoint files. Only used if "
                    "--ckpt-every > 0.")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="Save every N iters (0 disables). Default 0 ⇒ smoke "
                    "runs leave no artifacts; set ≥ 50 for long runs.")
    ap.add_argument("--resume", default=None,
                    help="Path to a .pt checkpoint to resume from. Restores "
                    "policy weights, Adam state, iter index, and history.")
    ap.add_argument("--split", default="training",
                    choices=["training", "validation", "testing"],
                    help="GPUDrive_mini split to train on. Default 'training' "
                    "(1000 scenes); 'validation' (150) stays held out for "
                    "eval_quick.py. Ignored when --data-root is given.")
    ap.add_argument("--data-root", default=None,
                    help="Explicit scene directory (overrides --split). For "
                    "the 10k dataset pass /data_new/training/group_0.")
    ap.add_argument("--inject-n", type=int, default=0,
                    help="Paper-style partial scene injection: # of the W "
                    "worlds refreshed with fresh scenes each cycle. 0 disables "
                    "(legacy fixed-batch). Paper ratio ≈ 2/3·W.")
    ap.add_argument("--inject-every", type=int, default=1,
                    help="Inject every N iterations (default 1 = every "
                    "iteration, paper's 'every batch').")
    a = ap.parse_args()

    if a.mode == "smoke":                       # M5b faithful short run
        print(f"M5b faithful run: iters={a.iters} scenes={a.scenes} "
              f"worlds={a.worlds} α={a.alpha} β={a.beta} "
              f"(exact per-agent KL, differentiable)")
        h, dw, dt = run(a.beta, a.iters, a.scenes, a.alpha, tag="",
                        n_worlds=a.worlds, ckpt_dir=a.ckpt_dir,
                        ckpt_every=a.ckpt_every, resume=a.resume, split=a.split,
                        data_root=a.data_root, inject_n=a.inject_n,
                        inject_every=a.inject_every)
        fin = all(np.isfinite([h[-1]['loss'], h[-1]['kl'], h[-1]['r_h']]))
        print(f"params changed (mean|Δw|)={dw:.2e} | finite={fin} | "
              f"{dt:.1f}s ({a.iters/dt:.2f} it/s)")
        print("VERDICT:", "M5b OK — faithful loop (exact per-agent Eq.5, "
              "differentiable), stable, π_θ updates"
              if (fin and dw > 0) else "M5b FAIL")
    else:                                       # M5c β-ablation + reactivity
        print(f"M5c ablation: {a.iters} iters/run, scenes={a.scenes}, "
              f"worlds={a.worlds}, α={a.alpha}, β∈{{0.0, {a.beta}}}")
        res = {}
        for b in (0.0, a.beta):
            h, dw, dt = run(b, a.iters, a.scenes, a.alpha,
                            tag="ABL ", n_worlds=a.worlds, split=a.split)
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
