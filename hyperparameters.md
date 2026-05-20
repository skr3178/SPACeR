# SPACeR — Hyperparameters & Architecture Nuances

Extracted from `SPACeR.pdf` (arXiv:2510.18060v2, "SPACeR: Self-Play Anchoring
with Centralized Reference Models", ICLR 2026). Main body + Appendix A.1–A.7.

---

## 1. Objective & Anchoring Hyperparameters

**Combined reward (Eq. 1):**
`r_t = r_t^task + α · r^humanlike(s_t, a_t)`

**Loss (Eq. 2):**
`L(θ) = L_PPO(θ; A[r]) − β · D_KL(π_θ(·|o_t) ‖ π_ref(·|s_t))`

- `r_humanlike(s_t,a_t) = log π_ref(a_t | s_t)` (Eq. 3) — dense per-timestep
  log-likelihood reward.
- KL computed **in closed form** over the shared discrete vocabulary (Eq. 5),
  single forward pass — no autoregressive sampling.
- **α (likelihood weight):** swept over {0.0, 0.001, 0.01, 0.1}.
- **β (KL alignment weight):** swept over {0.0, 0.01, 0.1, 1.0}; **default
  β = 0.01** for HR-PPO; tokenized ref can tolerate β = 1.0 with similar perf.
- Key finding: KL alignment term contributes most to realism; likelihood-only
  collapses entropy.

---

## 2. PPO Training Hyperparameters (Table A3)

| Parameter         | Value          | Description                          |
|-------------------|----------------|--------------------------------------|
| seed              | 42             | Random seed                          |
| total_timesteps   | 1,000,000,000  | Total env timesteps (1B)             |
| batch_size        | 131,072        | Timesteps collected per rollout      |
| minibatch_size    | 8,192          | Timesteps per optimization minibatch |
| learning_rate     | 3e-4           | Optimizer learning rate              |
| anneal_lr         | false          | LR annealing                         |
| gamma             | 0.99           | Discount factor                      |
| gae_lambda        | 0.95           | GAE parameter λ                      |
| update_epochs     | 4              | Optimization epochs per rollout      |
| norm_adv          | true           | Normalize advantages                 |
| clip_coef         | 0.2            | PPO policy clip coefficient          |
| clip_vloss        | false          | Clip value loss                      |
| vf_clip_coef      | 0.2            | Value function clipping coefficient  |
| ent_coef          | 0.0001         | Entropy coefficient                  |
| vf_coef           | 0.3            | Value loss coefficient               |
| max_grad_norm     | 0.5            | Gradient clipping (max L2 norm)      |

---

## 3. Self-Play Environment / Compute Config

- Simulator: **GPUDrive** (built on WOMD).
- Scenario: 9 s; initialize at **1 s**, simulate remaining **8 s**.
- Training set: **10k resampled WOMD scenarios**.
- **600 parallel worlds** (reference-free) → reduced to **300** for
  reference-model / HR-PPO training (~2× slowdown; no multi-GPU support in
  GPUDrive madrona backend).
- Up to **64 controlled agents** per rollout (shared decentralized policy).
- Single **NVIDIA A100 (80 GB, PCIe)** GPU; 1B env steps.
- Wall-clock: ~**24–48 hours** per run; results averaged over **5 seeds**.
- Memory-saving in ref-model setting: cap unique scenarios/batch at **200**;
  limit max map elements per agent **200 → 120**.
- Eval server: dual Intel Xeon Platinum 8358 (64 cores / 128 threads,
  2.6 GHz) + single A100.

---

## 4. Policy / Value Network Architecture

- **Late-fusion feedforward MLP** (per Kazemkhani / Cornelisse).
- Ego, partner, and road-graph features each embedded by a separate
  **two-layer MLP**, then concatenated → fused representation →
  **actor head + critic head**.
- **input embedding dim = 64**, **hidden dim = 128**, **dropout = 0.01**.
- **~65k parameters total** (decentralized; ~50× smaller than CAT-K's 3.2M).
- Decentralized: conditioned only on local observation; shared policy π_θ
  across all agents.

---

## 5. Observation Space

- Partially Observed Stochastic Game; agents act simultaneously.
- Ego-centric coordinate frame.
- Contents: nearby vehicles, lane geometry, goal points (optional), road
  features within **50 m radius**.
- **No temporal history**.
- All features normalized to **[−1, 1]**.

---

## 6. Action Space (Tokenized)

- Tokenized **Cartesian** trajectory action space (Philion / Wu / SMART),
  K-disk clustering.
- **K = 200 discrete tokens**.
- Each token = 0.1 s step, **horizon length 2** → **5 Hz action frequency**.
- Simulator runs at **10 Hz** while policy operates at **5 Hz** (frequency
  mismatch allowed by design).
- No explicit dynamics model — simulator advances per selected token.
- VRU / multi-agent: **separate action heads per agent type** + **one-hot
  agent-type indicator** in ego observation.

---

## 7. Reward Formulation & Weights

`r^task = w_goal·𝟙[Goal] − w_collided·𝟙[Collided] − w_offroad·𝟙[Offroad]
          + w_humanlike·r^humanlike`

- Default: **w_collided = w_offroad = 0.75**.
- Ablation grids: **w_goal ∈ {1.0, 0.5}**,
  **w_collided ∈ {0, −0.375, −0.75, +0.1}**,
  **w_offroad ∈ {0, −0.375, −0.75, +0.1}** (positive = adversarial
  stress-test).
- **goal-dropout**: train with/without goal conditioning while still giving
  terminal goal reward; once anchored to π_ref the explicit goal reward can
  be **removed entirely** with no performance loss.
- Reference model for KL comparison: **CAT-K**.

---

## 8. Reference Model (SMART / CAT-K) Architecture & Training

- Base SMART pretrained by BC on **16× A100 (80 GB)**, **10 epochs**, select
  ckpt at best val loss (**5e-4**).
- CAT-K closed-loop SFT: **6 additional epochs**, effective batch size **64**,
  LR **1e-9**.
- Final model: **3.2M parameters**, trained on full WOMD **~500k scenarios**.
- Sampling frequency **5 Hz** (vs SMART's original 2 Hz).
- Shares same backbone + action vocab (**K = 200**) as SMART/CAT-K baselines.
- Centralized: observes full state of all agents (privileged info,
  teacher–student style).
- Factorization (Eq. 4): conditional independence across agents —
  `p(a_t|a_<t,c) = ∏_i p(a_t^i|a_<t,c)`.
- Reference-model-quality study: SMART sizes **0.3M / 1M / 3M**; vocab sizes
  **100 / 200 / 400**.

---

## 9. HR-PPO Baseline

- Cornelisse & Vinitsky setup, adapted to tokenized action space.
- BC ref policy trained on **full WOMD** (not the 200-scenario subset).
- BC model ≈ **2× capacity** of self-play policy net; **val accuracy 92%**.
- `imitation` package; **60 epochs** on full dataset.
- KL regularization weight **β = 0.01** (larger β destabilizes; tokenized ref
  allows up to β = 1.0).
- Original-action-space variant required clamping min ref log-prob to
  **1e−20** (not needed with tokenized ref).
- Camera-ready uses **forward KL** (first submission used reverse KL, same as
  SPACeR).

---

## 10. WOSAC Evaluation Protocol

- Up to **128 agents/scenario**, simulated **8 s**, **32 multi-agent rollout
  samples**.
- NLL-based scoring (Eqs. 6–8), aggregated to composite realism.
- Evaluation on **2% validation subset**.
- Vehicles-only for main results; pedestrians/cyclists fixed to logged
  trajectories (except VRU experiments).
- Throughput at **5 Hz** single A100: SPACeR **211.8 ± 5.64** scenarios/s vs
  SMART **22.5 ± 0.01** (~10×).

---

## 11. Planner-Evaluation Variants (App. A.7)

- **22 self-play policies** (varying w_goal / w_collided / w_offroad grid
  above), shared late-fusion MLP, 1B steps, 600 worlds, 64 agents.
- **10 Frenet** + **10 IDM** rule-based variants.
- All planners: **dt = 0.1 s**, **wheelbase = 2.8 m**.

### Table A4 — Frenet-based Planner Variants

| Variant      | Description   | Speed (m/s) | Lateral W | Safety | Sampling (d,v,t) | Key feature           |
|--------------|---------------|-------------|-----------|--------|------------------|-----------------------|
| Baseline     | Balanced      | 0-30        | 10.0      | Medium | 10,5,3           | Standard config       |
| Aggressive   | High progress | 0-35        | 5.0       | Low    | 10,5,3           | Progress weight = 2.0 |
| Conservative | Safety-first  | 0-20        | 50.0      | High   | 10,5,3           | Collision penalty=5000|
| Smooth Rider | Comfort       | 0-30        | 20.0      | Medium | 10,5,3           | Jerk weight = 3.0     |
| Lane Keeper  | Centerline    | 0-30        | 100.0     | Medium | 15,5,3           | Lateral span = 1.5 m  |
| Wide Search  | Comprehensive | 0-30        | 10.0      | Medium | 20,10,7          | Large search space    |
| Fast Planner | Quick         | 0-30        | 10.0      | Medium | 5,3,2            | Reduced horizon       |
| Long Horizon | Strategic     | 0-30        | 10.0      | Medium | 10,5,3           | 40 horizon steps      |
| No Collision | Test baseline | 0-30        | 10.0      | None   | 10,5,3           | Collision disabled    |
| High Speed   | Highway       | 5-40        | 10.0      | Medium | 10,5,3           | Velocity span = 15    |

Frenet cost weights: lateral deviation (w_lateral), velocity tracking
(w_velocity), acceleration smoothness (w_acceleration), progress (w_progress),
jerk (w_jerk), collision penalty.

### Table A5 — IDM-based Planner Variants

| Variant         | Desc      | v0 (m/s) | s0 (m) | T (s) | Aggress. | Special           |
|-----------------|-----------|----------|--------|-------|----------|-------------------|
| IDM Baseline    | Standard  | 30       | 2.0    | 1.5   | 0.5      | Balanced          |
| IDM Conservative| Cautious  | 25       | 3.0    | 2.0   | 0.2      | Safety factor 1.5 |
| IDM Aggressive  | Dynamic   | 35       | 1.5    | 1.0   | 0.8      | Safety factor 0.9 |
| IDM Comfort     | Smooth    | 28       | 2.5    | 1.8   | 0.3      | Max jerk = 2.0    |
| IDM Highway     | High-speed| 40       | 3.0    | 1.2   | 0.6      | Perception = 100m |
| IDM City        | Urban     | 15       | 2.0    | 1.5   | 0.4      | Perception = 30m  |
| IDM Truck       | Heavy     | 25       | 4.0    | 2.0   | 0.3      | Length = 8.0 m    |
| IDM Emergency   | Urgent    | 40       | 1.5    | 0.8   | 0.9      | Max accel = 4.0   |
| IDM Adaptive    | Balanced  | 30       | 2.5    | 1.5   | 0.5      | Reaction = 0.2 s  |
| IDM Defensive   | Safety    | 25       | 4.0    | 2.5   | 0.1      | TTC = 3.0 s       |

IDM params: desired velocity (v0), min spacing (s0), safe time headway (T),
max acceleration (a), comfortable deceleration (b), aggressiveness (0.0–1.0).

### Table A6 — Key Configuration Parameters Comparison

**Frenet planner weights**

| Param            | Baseline | Aggressive | Conservative | Smooth | Lane Keeper |
|------------------|----------|------------|--------------|--------|-------------|
| Lateral (w_l)    | 10.0     | 5.0        | 50.0         | 20.0   | 100.0       |
| Velocity (w_v)   | 1.0      | 0.5        | 1.0          | 2.0    | 1.0         |
| Acceleration(w_a)| 1.0      | 1.0        | 3.0          | 5.0    | 1.0         |
| Progress (w_p)   | 1.0      | 2.0        | 1.0          | 1.0    | 1.0         |
| Jerk (w_j)       | 0.5      | 0.5        | 1.5          | 3.0    | 0.5         |

**IDM parameters**

| Param            | Baseline | Aggressive | Conservative | Smooth |
|------------------|----------|------------|--------------|--------|
| Desired vel (v0) | 30.0     | 35.0       | 25.0         | 28.0   |
| Min spacing (s0) | 2.0      | 1.5        | 3.0          | 2.5    |
| Time headway (T) | 1.5      | 1.0        | 2.0          | 1.8    |
| Max accel (a)    | 2.0      | 3.0        | 1.5          | 1.5    |
| Comfort decel (b)| 3.0      | 4.0        | 2.0          | 2.0    |

Note: all planners use dt = 0.1 s time step and wheelbase = 2.8 m.

---

## Key Architecture Nuances

1. **RL-first, not pretrain-then-finetune** — self-play is the foundation; the
   IL model is injected only as a reward / KL provider.
2. **Aligned discrete action space** between π_θ and π_ref enables the
   **closed-form KL** (Eq. 5) and likelihood reward with a single forward
   pass — no online tokenization, no autoregressive sampling during training.
3. **Asymmetric setup**: π_θ decentralized + local obs + no history; π_ref
   centralized + full scene context (privileged-info / teacher–student
   analogy).
4. **Per-agent, per-timestep credit assignment**: π_ref gives a distinct
   distribution for each agent at each timestep, addressing MARL credit
   assignment vs sparse trajectory-level rewards.
5. **Frequency decoupling**: policy 5 Hz, simulator 10 Hz — possible because
   actions are Cartesian tokens, not dynamics controls.
6. **goal-dropout**: removes reliance on explicit goal inputs; explicit goal
   reward removable once anchored.
7. **VRU specialization**: separate per-type action heads + one-hot type
   token; reference KL loss is the single most important component for VRU
   realism (ablation A1).
8. **Reference model as soft prior, not imitation target**: SPACeR realism
   stays clustered ~0.73 even with a weak 0.3M / 0.636-realism reference —
   closed-loop interaction lets policies exceed the reference.
