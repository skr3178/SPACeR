# SPACeR Training Pipeline — Architecture

Reference visualization: [viz_pi_ref.gif](viz_pi_ref.gif) and
[viz_pi_ref.png](viz_pi_ref.png).
GIF / PNG panel mapping:

- **Panel A** = GT log (WOMD)  — eval only, never enters training
- **Panel B** = π_ref (CAT-K `clsft_E9`) — frozen scorer
- **Panel C** = π_θ closed-loop rollout — the thing being trained

---

## Pipeline

```
┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐
┊  HISTORICAL — NOT IN OUR PIPELINE † (done once by CAT-K authors;  ┊
┊  output is the file `checkpoints/clsft_E9.ckpt` we just load)     ┊
┊                                                                   ┊
┊    WOMD ~500k scenes ──BC + CAT-K closed-loop SFT──> π_ref        ┊
┊                                                       ▲           ┊
┊                                              FROZEN — 7 M params  ┊
┊                                              "Panel B" in the GIF ┊
└┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
                                                       │
                                                       │ load .ckpt, no_grad
                                                       ▼
╔═══════════════════════════════════════════════════════════════════╗
║  ONLINE TRAINING LOOP  (paper: 1B env steps, ~24–48 h on A100)    ║
║                                                                   ║
║   GPUDrive scene init (1 s history from WOMD mini)                ║
║              │                                                    ║
║              ▼                                                    ║
║   ┌──────────────────────────────────────┐                        ║
║   │  GPUDrive sim — 64 agents, all using │  ← SELF-PLAY:          ║
║   │  the SAME π_θ; 8 s closed-loop       │    one shared policy   ║
║   │  rollout                             │    drives all 64       ║
║   └────┬─────────────────────────┬───────┘                        ║
║        │ states + chosen tokens  │  this is "Panel C" in the GIF  ║
║        │                         │                                ║
║        │ r_task                  │ scene + per-step states        ║
║        │ collide / off-road      ▼                                ║
║        │                ┌────────────────────────┐                ║
║        │                │  π_ref forward — ONE   │                ║
║        │                │  pass over rollout,    │                ║
║        │                │  no grad, frozen       │                ║
║        │                └──────────┬─────────────┘                ║
║        │                           │ per-(agent,step) distributions
║        │                           │ over 2048 tokens             ║
║        │                  ┌────────┴────────┐                     ║
║        │                  ▼                 ▼                     ║
║        │       r_h = log π_ref(a_C|s)   KL(π_θ ‖ π_ref)           ║
║        │       (Eq. 3 — α=0, logged)    (Eq. 5, closed-form)      ║
║        │                  │                 │                     ║
║        ▼                  ▼                 ▼                     ║
║   ┌──────────────────────────────────────────────────┐            ║
║   │  Loss = − L_PPO( r_task + α·r_h )  +  β · KL     │            ║
║   └────────────────────────┬─────────────────────────┘            ║
║                            ▼                                      ║
║              ┌────────────────────────┐                           ║
║              │  PPO update on π_θ     │  (~304 k MLP*)            ║
║              │  π_ref NEVER changes   │                           ║
║              └───────────┬────────────┘                           ║
║                          │                                        ║
║                          └────── π_θ updated, next iter ──┐       ║
║                                                           │       ║
║   ◄───────────────────────────────────────────────────────┘       ║
╚═══════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════╗
║  EVALUATION ONLY  (WOSAC; no training)                            ║
║                                                                   ║
║      trained π_θ rolls in GPUDrive ─┐                             ║
║                                     ├─► WOSAC NLL → realism       ║
║      GT logged 8 s future ──────────┘                             ║
║                                     │                             ║
║                                     └── "Panel A" used HERE,      ║
║                                         never during training     ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Key things to read off the diagram

1. **WOMD enters only twice — both outside the training loop:**
   (a) once to pre-train π_ref offline,
   (b) once at eval as ground truth.
   During training, the loop never sees logged trajectories.
2. **One trajectory per iteration.** GPUDrive produces it (Panel C);
   π_ref reads its states (Panel B's role) and outputs *distributions*,
   not trajectories.
3. **Self-play** = the same π_θ drives all 64 agents — that's the box
   at the top of the loop. They learn against each other.
4. **Only π_θ updates.** π_ref is loaded once and frozen for the entire run.
5. **Panel A is invisible to training.** It exists only as a yardstick at
   WOSAC eval time.

---

## Component sizes (this codebase)

| Symbol | Role | Source | Params |
|---|---|---|---|
| **GPUDrive** | physics/sim engine (environment) | installed package `/gpudrive/` in the image | — |
| **π_θ** | trainable policy (online PPO target) | `policy_token.TokenPolicy` (late-fusion MLP, 2048-token actor head) | **303,681** (≈304 k)* |
| **π_ref** | frozen human-likeness prior | `clsft_E9.ckpt` → `SMARTDecoder` | ~7 M |
| `policy_S10_000_02_27` | unused here; GPUDrive baseline only — we do **not** warm-start (SPACeR is RL-first; π_θ is random-init) | GPUDrive repo's checkpointed self-play policy | ~65 k (91-action head) |

\* Paper-equivalent π_θ is **~65 k** (same backbone + ~200-token actor head).
Our backbone is **byte-identical** (39,360 params); the **~4.7× delta**
(303,681 vs 65,289) is *entirely* the wider actor head (`Linear(128, 2048)` =
264,192 params vs ~25,800) — the locked consequence of anchoring to public
`clsft_E9` (vocab 2048) so the closed-form KL is token-aligned. The trainable
*backbone* — the part that encodes the scene — matches the paper exactly.
See § *π_θ architecture (detailed)* below and `STAGE_PLAN.md` S2.6 for the
optional vocab-coarsening path.

## π_θ architecture (detailed)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Late-fusion MLP: 3 per-modality encoders → max-pool → fuse →        │
│                   actor (2048-way) + critic (scalar)                 │
│  Source: policy_token.TokenPolicy                                    │
│        → gpudrive.networks.late_fusion.NeuralNet(action_dim=2048)    │
└──────────────────────────────────────────────────────────────────────┘

input obs  [N, 2984]   (ego_state ‖ partner_obs(×63) ‖ road_map(×?))
   │
   ├── ego_embed         Linear(ego_dim → 64) → tanh → Linear(64 → 64)
   ├── partner_embed     Linear(partner_dim → 64) → tanh → Linear(64 → 64)
   │                       then max-pool across partner agents      → [N, 64]
   └── road_map_embed    Linear(roadgraph_dim → 64) → tanh → Linear(64 → 64)
                          then max-pool across road points          → [N, 64]
                                                │ concat
                                                ▼
                                       [N, 192]   (= 3 × 64)
                                                │
                            shared_embed  Linear(192 → 128) → tanh → Dropout(0.01)
                                                │
                                                ▼
                                          hidden  [N, 128]
                                          ┌───────┴───────┐
                                          ▼               ▼
                            actor   Linear(128 → 2048)    critic  Linear(128 → 1)
                            ← 2048-way categorical        ← state value
                              (SPACeR head; baseline was 91)
```

**Parameter breakdown (verified `policy.num_params()`):**

| Component | Shape | Params |
|---|---|---|
| Backbone (ego_embed + partner_embed + road_map_embed + shared_embed) | — | **39,360** |
| Actor head `Linear(128, 2048)` | 128·2048 + 2048 | **264,192** |
| Critic head `Linear(128, 1)` | 128·1 + 1 | **129** |
| **Total π_θ** | | **303,681** |
| Paper-equivalent (same backbone + 200-token actor) | | 65,289 ≈ "~65 k" |

**Hyperparameters:** `input_dim=64`, `hidden_dim=128`, `dropout=0.01`
(paper A.3 — `NeuralNet`'s own default is 0.0, we override; Test 18 close),
`act_func="tanh"`, `max_controlled_agents=64`, `obs_dim=2984`. Built from
`NeuralNet(action_dim=2048, dropout=0.01, config={...})`.

**Initialisation:** PufferLib `std=0.01` on the actor ⇒ near-uniform output at
init (verified M2: init entropy = **7.625 ≈ ln 2048**; no degenerate collapse).

**Optimizer:** Adam, `lr=3e-4` (default in `train_spacer.py`).

## Loss summary (Eqs. 1, 2, 3, 5) — **Variant 4 chosen as default**

Paper Table A2 ablation conclusions (WOSAC validation):
**KL is the only load-bearing anchoring term**; LLH (Eq. 3 reward) and goal
reward are both droppable without composite loss. We adopt **Variant 4
("KL + r_inf")** — Table A2 best composite (0.74) with the fewest tunables.

```
r_inf      = − w_coll·𝟙[collision] − w_off·𝟙[off-road]      # safety only
KL         = Σ_a π_θ(a|o) · log( π_θ(a|o) / π_ref(a|s) )      # Eq. 5
Loss(θ)    = − L_PPO( r_inf )  +  β · KL                      # Eq. 2 (reduced)

  α  = 0      ⇒ drop LLH (Eq. 3) — Table A2: no composite gain vs KL alone
  w_goal = 0  ⇒ drop goal reward  — Table A2 caption: "goals unnecessary"
  w_coll = 0.75   w_off = 0.75
  β  ≈ 0.01–0.1   ← the single anchoring scalar to tune
```

This is wired in `train_spacer.py` (`build_env`: `goal_achieved_weight=0`,
`collision_weight=-0.75`, `off_road_weight=-0.75`, `reward_type=
"weighted_combination"`) and `run(... alpha=0.0)`.

### Variant table (for ablation reference)

| Variant | Composite ↑ | minADE ↓ | Tunables | Verdict |
|---|---|---|---|---|
| 1. r_task only | 0.70 | 14.43 | w_*  | PPO baseline / smoke only |
| 2. Goal + LLH | 0.69 | 21.05 | α, w_goal | **AVOID** — entropy collapse |
| 3. Goal + KL | 0.73 | 4.08 | β, w_goal | only if minADE is target |
| **4. KL + r_inf** ⭐ | **0.74** | 4.73 | β, w_coll, w_off | **DEFAULT** — Pareto best composite, fewest hyperparams |
| 5. KL + r_inf + LLH | 0.74 | 4.68 | α, β, w_coll, w_off | tied composite; α not worth tuning |

---

## Algorithm 1 — as implemented

> `algorithm.png` is the paper's screenshot of Algorithm 1. This is the
> editable, code-mapped equivalent — what `spacer/train_spacer.py` actually
> runs after the Test 18 PPO port + Test 22 rollout-accumulation refactor.
> Variant 4 (α=0, w_goal=0).

```
Require: π_ref = clsft_E9 (frozen SMARTDecoder)              load_ref()
Require: β (KL weight); w_coll = w_off = 0.75; α = 0         Variant 4
Require: K (rollout-accumulation factor, --accum-k)          default 1
 1  init π_θ — late-fusion MLP, 2048-token actor + critic    TokenPolicy
 2  for each iteration (--iters):                            run() loop
 3    buffer ← []                                            spacer_iteration()
 4    for k = 1 .. K:                                        K micro-rollouts
 5       if k > 0 or do_inject:  inject_scenes()             paper-style refresh
 6       ROLLOUT  — W worlds, T token-steps;                 rollout()
            per (agent, step): obs, tok, logp, value,
            logits, r_task
 7       π_ref scores the FULL rolled scene — one fwd        score_ref()
            → next_token_logits = p(a_t | a_<t, c)
 8       r_humanlike = log π_ref(a_t | a_<t, c)    [Eq.3]    (logged; α=0)
 9       r_task = −0.75·𝟙[coll] − 0.75·𝟙[off]      [V4]      env weighted_combination
10       D_KL = Σ_a π_θ(a|o) log(π_θ/π_ref)        [Eq.5]
11       A[r] ← GAE(γ=0.99, λ=0.95)                          _gae()
12       flatten to per-sample CPU buffer, append             _collect()
13    PPO UPDATE — 4 epochs × 16 minibatches:                _ppo_update()
         For each minibatch in concat(buffer):
14         newlp, entropy, value, log_probs ←                policy.forward_with_logits()
                policy.forward_with_logits(o, a)             ONE backbone fwd (D)
15         kl_mb = Σ_a π_θ(a)·log(π_θ/π_ref) [Eq.5, recomp]
16         loss = pg − ent_coef·H + vf_coef·v_loss + β·kl_mb [Eq.2]
17         Adam step (lr 3e-4, grad-norm clip 0.5, seed 42)
24  return π_θ
```

**Optimizer = the paper's PPO.** `L_PPO` is the GPUDrive PufferLib PPO
(`gpudrive/integrations/puffer/ppo.py`) ported into `_ppo_update`, with
**all 13 algorithmic Table A3 hyperparameters matched verbatim** (γ, λ, clip,
vf_coef, ent_coef, max_grad_norm, update_epochs, norm_adv, clip_vloss,
vf_clip, lr, anneal_lr, seed). Pre-Test-18 this was a compact REINFORCE-style
PG; it is now real PPO. The closed-form KL anchor (Eq. 5) is recomputed
per-minibatch (π_θ moves each update; π_ref frozen) and added to the loss.

**Rollout accumulation (Test 22).** `spacer_iteration` now wraps K
micro-rollouts into one PPO update, so per-update samples = K × W × T ×
(cmask agents). K=1 reproduces the prior single-rollout behaviour exactly
(back-compat). K=46 gives **~126,576 samples per update — 96% of the
paper's `batch_size=131,072`** for our 2 Hz cadence. Buffers live on
CPU between micro-rollouts to keep VRAM available for the PPO forward;
each minibatch is moved to GPU on demand.

**Single-forward PPO (Test 22, "D" fix).** `TokenPolicy.forward_with_logits`
computes `(newlp, entropy, value, log_probs)` in *one* backbone+actor pass.
The prior pattern called the policy twice per minibatch (once for newlp/
entropy/value, again for raw logits used by the closed-form KL); replacing
it with a single forward roughly halves PPO-step activation memory,
mathematically identical.

**Road-context cap (Test 22, "E" fix).** `build_env` now sets
`roadgraph_top_k=120` on `EnvConfig` (default 200). This is the paper's own
§A.3 trick *"we limit the maximum number of map elements per agent from
200 to 120 to reduce GPU memory consumption in the reference-model
setting."* Shrinks `obs_dim` (input) and Madrona per-agent road state.

**Deviations from the paper's Algorithm 1 — all scale, none algorithmic:**

| Aspect | Paper | Ours | Reason |
|---|---|---|---|
| parallel worlds | 300 | ≤48 | RTX 3060 12 GB ceiling (Test 17) |
| total env-steps | 1×10⁹ | `--iters`·K·W·T bound (Test 22: ~6×10⁶) | compute ceiling — ~160× less |
| batch_size (samples/update) | 131,072 | **K=46: 126,576 (96% of paper)** | rollout-accum (Test 22) |
| minibatch (samples/step) | 8,192 | **K=46: ~7,911 (96% of paper)** | K=W=24, T=18, ÷16 minibatches |
| β | 0.01 | 0.1 | Test 13 — 0.01 degenerate at our budget; both in paper's robust band |
| control cadence | 5 Hz | 2 Hz | `clsft_E9` native tokenisation |
| road-context per agent | 120 | 120 | paper §A.3 verbatim (Test 22 "E") |
| vocab (action space) | ~200 | 2,048 | locked by public `clsft_E9` checkpoint |

The SPACeR *algorithm* (Eqs. 1/2/3/5, PPO, Variant 4 reward) is faithfully
reproduced; per-update batch is now also paper-faithful (K=46). What
remains is **total compute budget** (~160× short) and the **vocab
mismatch** (2,048 vs paper's ~200).

### Pipeline status — Tests 20/21/22

The full loop has been executed across three increasingly paper-faithful
attempts (`test.md` Tests 20, 21, 22):

- **Test 20** — Variant 4 long run, fixed 24 scenes (no injection). Degenerate
  optimum: goal_rate 0.019, minADE ≈ 25 m vs teacher ≈ 4.5 m.
- **Test 21** — β-sweep {0.01, 0.10, 1.00} with paper-style full-resample
  injection on the new 10k dataset. **All three degenerate** (goal ~0.02,
  minADE ~27–30 m). Anchor strength and injection ruled out.
- **Test 22** — paper-batch K=46 (= 126,576 samples/update, 96% of paper)
  with the single-forward and `roadgraph_top_k=120` fixes. **No improvement
  vs Test 21 b=0.10** (goal 0.014, minADE 28.2 m). **Per-update batch size
  also ruled out.**

**Cumulative empirical conclusion:** the persistent degenerate optimum is
**not** caused by dataset size, control cadence, scene injection, anchor
strength (β), per-update batch size, batch quality, or road-context size —
all have been ruled out by controls. The two remaining suspects are
**total env-step budget** (paper 10⁹ vs ours ~10⁶–10⁷, ~160× short) and
the **π_ref vocab mismatch** (paper ~200 tokens vs our public-`clsft_E9`
2,048-token vocab, which may make the closed-form KL anchor too soft per
token to constrain closed-loop trajectories regardless of β).

---

## Datasets — what we need, what we don't

| Asset | Used for | Source / status |
|---|---|---|
| **`clsft_E9.ckpt`** | frozen π_ref (Panel B) | ✓ `checkpoints/clsft_E9.ckpt` (71 MB), shipped |
| **GPUDrive_mini** (~1 k train / 150 val / 150 test) | early tests (12–20) | ✓ `/gpd/data/processed/` |
| **GPUDrive 10k training + 941 validation** (HF `EMERGE-lab/GPUDrive`) | **default training + eval set** for Tests 21/22 | ✓ host `/home/skr/gpudrive_data/`; mounted into container as `/data_new/` (Test 21 onward). CLI: `--data-root /data_new/training/group_0` for training, `--data-root /data_new/validation` for eval. |
| **WOMD raw ~500 k protos** | only if **retraining π_ref** from scratch | ✗ **not needed for our pipeline**. See † below. |

### † Footnote — when WOMD 500 k *would* be needed

The dashed "HISTORICAL" box above is the work the CAT-K authors did once to
produce `clsft_E9.ckpt`. We never run it. The only paths that re-enter that
box are upgrades to the reference model's cadence:

- **Tier 1 — 5 Hz distillation** (warm-start from `clsft_E9`): does not
  strictly require raw WOMD; can use `clsft_E9` rollouts as the synthetic
  teacher data (see [Student_Teacher_Distillation.md](Student_Teacher_Distillation.md)).
- **Tier 2 — 5 Hz from scratch** (paper's A.4 path): **requires the full
  WOMD ~500 k**, a Waymo license, ~1 TB download, and ~150–1000 A100-hours.
  Avoid unless Tiers 0 and 1 both fail.

Trigger condition: M4 reactivity gate fails at 2 Hz. Until then, the box
stays dashed and the ~500 k WOMD scenes are not in scope.
