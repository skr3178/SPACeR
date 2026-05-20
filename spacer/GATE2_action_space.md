# Gate #2 — Action-Space Alignment (scoping/design)

**Question scoped:** how does π_θ produce a 2048-token categorical, and how does
GPUDrive advance the sim from a chosen trajectory token?

## Feasibility verdict: ✅ CONFIRMED (no simulator changes needed)

GPUDrive natively supports 4 dynamics models (`base_env.py:103`): `classic`,
`bicycle`, `delta_local`, `state`. Two are directly usable for token-driven
motion, and **both accept continuous values** (not just the coarse config grids)
— verified in `env_torch.py:_apply_actions` / `_copy_actions_to_simulator`:

| Dynamics | Continuous action accepted | Vector |
|---|---|---|
| `delta_local` | 3-D action, last dim ≠ 1 → used as raw values | `(dx, dy, dyaw)` local |
| **`state`** (recommended) | always raw values → `sim.action_tensor()[:,:,:10]` | `(x,y,z,yaw,vx,vy,vz,ωx,ωy,ωz)` global |

→ `state` lets an agent **exactly execute the decoded token trajectory**, so
π_θ's realized motion *is* the token. That makes Eq. 5 KL and Eq. 3 reward exact
and consistent with how π_ref tokenizes. This was the one open feasibility risk
(coarse grids) — **resolved**.

## Architecture

### π_θ — token policy
- Reuse the GPUDrive late-fusion MLP backbone (Test 2 baseline, ~65k params);
  **replace the 91-way (13×7 accel/steer) head with a 2048-way categorical**
  over the agent token vocab, conditioned on GPUDrive local obs `o_t`.
- Per agent type (veh/ped/cyc each 2048); vehicles-only first.
- Output: `π_θ(token | o_t)` — categorical over the **same 2048** as π_ref.

### Token → pose decode (reuse CAT-K, no reimplementation)
- Vocab: `TokenProcessor.agent_token_all_{veh,ped,cyc}` = `(2048, 6, 4, 2)`
  local contours (6 sub-steps × 4 bbox corners × xy), 0.5 s / 10 Hz.
- Given current `(pos, head)`: `transform_to_global(token_traj, pos_now, head_now)`
  → 6 global sub-step contours. Per sub-step: `pos = contour.mean(1)`,
  `head = atan2(corner0 − corner3)` — **exactly** the `prev_pos`/`prev_head`
  update at `token_processor.py:254-258`; `cal_polygon_contour` at
  `src/smart/utils/rollout.py:23`. Velocity = finite-diff of sub-poses (already
  have `finite_diff_velocity` in `gpudrive_to_smart.py`).

### Step structure — token cadence = **2 Hz** (locked by the public checkpoint)

`clsft_E9` is tokenized at `shift=5` @ 10 Hz → **one token = 0.5 s = 2 Hz
decisions**. The paper's A.4 uses 5 Hz, but that required *their own*
re-tokenized/retrained reference; the fixed public checkpoint cannot do 5 Hz
without retraining π_ref, so our SPACeR loop runs at the checkpoint's native
**0.5 s / 2 Hz** cadence (a documented Strategy-A consequence: coarser control
than the paper, acceptable).

Every 0.5 s (5 sim steps):
1. π_θ observes `o_t` → samples token `a_t ∈ [0,2048)`.
2. Decode `a_t` → 5 global sub-poses (+vel) via CAT-K geometry.
3. Drive 5 sim sub-steps with continuous `state` actions along the sub-poses.
4. π_ref via the (live, gate #1) adapter → `π_ref(·|s_t)` over the same 2048
   at the same 0.5 s / 2 Hz cadence.

### Eq. 5 / Eq. 3 — now exact
Both π_θ and π_ref are categoricals over the identical 2048 tokens at each
0.5 s decision step:
- **Closed-form KL (Eq. 5):** `Σ_{a=1..2048} π_θ(a) · log(π_θ(a)/π_ref(a))` —
  direct sum, no sampling. *This is the discriminative signal Test 5 identified
  as the real one — now computable.*
- **Likelihood reward (Eq. 3):** `r_humanlike = log π_ref(a_t | s_t)` — index
  π_ref logits at the executed token.
- **Loss (Eq. 2):** `L = L_PPO(θ; A[r]) − β·D_KL`; `r = r_task + α·r_humanlike`
  (Eq. 1). PPO via PufferLib (already in `catk-spacer` image; GPUDrive has a
  PPO baseline in `integrations/puffer/ppo.py` to adapt).

## Reuse map

| Need | Source | Status |
|---|---|---|
| Sim + continuous `state` dynamics | GPUDrive (`dynamics_model="state"`) | exists |
| π_ref categorical over 2048 from live state | `gpudrive_to_smart.py` + adapter | ✅ built (gate #1) |
| Token→global-pose decode | CAT-K `transform_to_global`, `cal_polygon_contour`, contour→pose | exists, reuse |
| PPO trainer | PufferLib / GPUDrive `integrations/puffer/ppo.py` | exists, adapt |
| **π_θ 2048-token head** | new (head swap on late-fusion MLP) | NEW |
| **token→state action driver** (decode + 5 sub-step rollout) | new (thin, reuses CAT-K geometry) | NEW |
| **Eq. 3/5 + Eq. 2 loss assembly** | new (small) | NEW |
| Decision-step ↔ π_ref token-step alignment | new (mirror tokenize_agent's 5-step windowing) | NEW |

## Build milestones (incremental, each testable)

1. **Token→state driver**: decode a fixed token sequence, drive `state`
   dynamics, verify an agent follows the intended trajectory (sanity: feed the
   GT tokens → agent ≈ logged path).
2. **π_θ head swap**: late-fusion MLP with 2048 categorical head; random-init
   forward on GPUDrive obs → valid distribution.
3. **Eq. 5 KL + Eq. 3** wired at the 0.5 s cadence; re-run the Test-5 contrast
   — now KL *should* be strongly discriminative (the real validation deferred
   from Test 5).
4. **PPO+KL loop** (Eq. 2), scaled-down on the 3060 (few worlds, short run);
   confirm learning signal / no divergence.
5. Scale tuning.

## Open items / risks (honest)

- **Decision-step alignment**: π_ref adapter emits `next_token_logits[A,16,2048]`
  for token-steps (10→15)…(85→90); π_θ acts at the same cadence — must index
  the matching step (mirror `tokenize_agent`'s 5-step windowing). Low risk,
  needs care.
- **Throughput**: adapter + π_ref forward now run *inside* the RL loop every
  0.5 s across worlds. Batched, no-grad; still the main 3060 bottleneck →
  milestone 4/5 scale tuning.
- **Non-controlled agents**: in self-play all agents are π_θ-controlled;
  uncontrolled replay logs (existing GPUDrive behavior).
- **Agent-type vocab**: 2048 per veh/ped/cyc; start vehicles-only.
- **`state` collisions/validity**: setting absolute pose ignores physics —
  acceptable (tokens encode realistic motion); keep `collision_behavior` as in
  the paper's reward.

## Bottom line

Paper-faithful action-space alignment is **feasible with stock GPUDrive**
(`state` dynamics, continuous) + **reuse of CAT-K decode geometry**. New code is
modest and well-bounded: a 2048 policy head, a token→state driver, and the
Eq. 2/3/5 assembly. Milestone 3 finally validates the *dominant* π_ref signal
(closed-form KL) that Test 5 showed token-NLL alone could not.
