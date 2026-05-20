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
║        │ goal/collide/off-road   ▼                                ║
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
║        │       (Eq. 3, dense reward)    (Eq. 5, closed-form)      ║
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
| `policy_S10_000_02_27` | unused here; baseline / optional warm-start | GPUDrive repo's checkpointed self-play policy | ~65 k (91-action head) |

\* Paper's π_θ is **~65 k** with a 200-token actor head. Our backbone is
**byte-identical** (39,360 params); the 5× delta is entirely the wider actor
head (`Linear(128, 2048)` = 264,192 params) — the locked consequence of using
public `clsft_E9` (vocab 2048). See § *π_θ architecture (detailed)* below and
`STAGE_PLAN.md` S2.6 for the optional vocab-coarsening path.

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
                            shared_embed  Linear(192 → 128) → tanh → Dropout(0)
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

**Hyperparameters (defaults):** `input_dim=64`, `hidden_dim=128`, `dropout=0.0`,
`act_func="tanh"`, `max_controlled_agents=64`, `obs_dim=2984`. Built from
`NeuralNet(action_dim=2048, config={"reward_type":"weighted_combination", "vbd_in_obs":False})`.

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

## Datasets — what we need, what we don't

| Asset | Used for | Source / status |
|---|---|---|
| **`clsft_E9.ckpt`** | frozen π_ref (Panel B) | ✓ `checkpoints/clsft_E9.ckpt` (71 MB), shipped |
| **GPUDrive-processed scenes** | scene inits for the training loop (Panel C rollouts) and WOSAC eval | ✓ GPUDrive_mini (~1 k) in `/gpd/data/processed/`. Paper uses ~10 k via HF `EMERGE-lab/GPUDrive`. |
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
