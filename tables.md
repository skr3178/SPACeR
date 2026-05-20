# Tables — SPACeR: Self-Play Anchoring with Centralized Reference Models

Extracted from `SPACeR.pdf` (arXiv:2510.18060v2, ICLR 2026).

---

## Table 1: Results on the WOSAC Validation Set (p.7)

Our proposed method outperforms other self-play approaches across all realism metrics, while achieving ~10× higher throughput than imitation-learning (shaded) methods, with competitive performance and lower collision/off-road rates. Throughput is measured in scenarios/sec at 5 Hz on a single A100 GPU.

| Method | Composite ↑ | Kinematic ↑ | Interactive ↑ | Map ↑ | minADE ↓ | Collision ↓ | Off-road ↓ | Throughput ↑ |
|---|---|---|---|---|---|---|---|---|
| PPO    | 0.710 ±0.01 | 0.327 ±0.01 | 0.751 ±0.01 | 0.875 ±0.00 | 12.725 ±2.53 | 0.038 ±0.005 | 0.053 ±0.00 | 211.8 ±5.6 |
| HF-PPO | 0.716 ±0.00 | 0.341 ±0.00 | 0.756 ±0.00 | 0.880 ±0.00 | 12.254 ±1.02 | 0.044 ±0.006 | 0.053 ±0.00 | 211.8 ±5.6 |
| **SPACeR** | **0.741** ±0.00 | **0.411** ±0.00 | **0.779** ±0.01 | 0.880 ±0.00 | 4.101 ±0.09 | 0.036 ±0.01 | 0.056 ±0.00 | 211.8 ±5.6 |
| SMART  | 0.720 | 0.450 | 0.725 | 0.870 | 1.840 | 0.17 | 0.13 | 22.5 ±0.0 |
| CAT-K  | 0.766 | 0.490 | 0.792 | 0.890 | 1.470 | 0.06 | 0.09 | 22.5 ±0.0 |

*(SMART and CAT-K are imitation-learning baselines — shaded rows in the original.)*

---

## Table 2: VRU realism metrics on WOSAC (pedestrians and cyclists) (p.8)

SPACeR outperforms PPO and HR-PPO by a large margin, achieving substantial gains across all realism metrics and minADE.

| Method | Composite ↑ | Kinematic ↑ | Interactive ↑ | Map ↑ | minADE ↓ |
|---|---|---|---|---|---|
| PPO            | 0.648 | 0.242 | 0.683 | 0.835 | 7.712 |
| HR-PPO         | 0.668 | 0.285 | 0.700 | 0.847 | 7.014 |
| **SPACeR (Ours)** | **0.729** | **0.413** | **0.762** | **0.866** | **2.066** |

---

## Table A1: Ablation study on key components of SPACeR adapted to VRU simulation (p.14)

Each row removes one design choice. Metrics are reported over VRU target agents only.

| Ablation | Composite ↑ | Kinematic ↑ | Interactive ↑ | Map-based ↑ | minADE ↓ |
|---|---|---|---|---|---|
| Full Model            | **0.729** | **0.413** | 0.762 | **0.866** | **2.066** |
| – goal-reaching weight | 0.728 | 0.405 | **0.769** | 0.859 | 2.295 |
| – multi-action head    | 0.685 | 0.323 | 0.742 | 0.818 | 3.416 |
| – reference KL loss    | 0.607 | 0.222 | 0.626 | 0.804 | 12.844 |

---

## Table A2: Ablations on WOSAC (validation) (p.14)

r_inf = infraction penalties (off-road, collision); LLH = log-likelihood reward; KL alignment essential for realism; goals unnecessary.

| Variant | Composite ↑ | minADE ↓ |
|---|---|---|
| r_task only       | 0.70 | 14.43 |
| Goal + LLH        | 0.69 | 21.05 |
| Goal + KL         | 0.73 | **4.08** |
| KL + r_inf        | **0.74** | 4.73 |
| KL + r_inf + LLH  | **0.74** | 4.68 |

---

## Table A3: PPO Training Hyperparameters (p.16)

| Parameter | Value | Description |
|---|---|---|
| seed            | 42            | Random seed. |
| total_timesteps | 1,000,000,000 | Total number of environment timesteps. |
| batch_size      | 131,072       | Timesteps collected per rollout. |
| minibatch_size  | 8,192         | Timesteps per optimization minibatch. |
| learning_rate   | 3e-4          | Optimizer learning rate. |
| anneal_lr       | false         | Learning rate annealing. |
| gamma           | 0.99          | Discount factor. |
| gae_lambda      | 0.95          | GAE parameter λ. |
| update_epochs   | 4             | Optimization epochs per rollout. |
| norm_adv        | true          | Normalize advantages. |
| clip_coef       | 0.2           | PPO policy clip coefficient. |
| clip_vloss      | false         | Clip value loss. |
| vf_clip_coef    | 0.2           | Value function clipping coefficient. |
| ent_coef        | 0.0001        | Entropy coefficient. |
| vf_coef         | 0.3           | Value loss coefficient. |
| max_grad_norm   | 0.5           | Gradient clipping (max L2 norm). |

---

## Table A4: Summary of Frenet-based Planner Variants (p.19)

Sampling notation: (d,v,t) represents (lateral samples, velocity samples, time samples).

| Variant | Description | Speed (m/s) | Lateral Weight | Safety Focus | Sampling (d,v,t) | Key Features |
|---|---|---|---|---|---|---|
| Baseline      | Balanced       | 0–30 | 10.0  | Medium | 10,5,3  | Standard configuration |
| Aggressive    | High progress  | 0–35 | 5.0   | Low    | 10,5,3  | Progress weight = 2.0 |
| Conservative  | Safety-first   | 0–20 | 50.0  | High   | 10,5,3  | Collision penalty = 5000 |
| Smooth Rider  | Comfort        | 0–30 | 20.0  | Medium | 10,5,3  | Jerk weight = 3.0 |
| Lane Keeper   | Centerline     | 0–30 | 100.0 | Medium | 15,5,3  | Lateral span = 1.5m |
| Wide Search   | Comprehensive  | 0–30 | 10.0  | Medium | 20,10,7 | Large search space |
| Fast Planner  | Quick          | 0–30 | 10.0  | Medium | 5,3,2   | Reduced horizon |
| Long Horizon  | Strategic      | 0–30 | 10.0  | Medium | 10,5,3  | 40 horizon steps |
| No Collision  | Test baseline  | 0–30 | 10.0  | None   | 10,5,3  | Collision disabled |
| High Speed    | Highway        | 5–40 | 10.0  | Medium | 10,5,3  | Velocity span = 15 |

---

## Table A5: Summary of IDM-based Planner Variants (p.20)

Aggressiveness factor: 0.0 (very conservative) → 1.0 (very aggressive). TTC: time-to-collision threshold.

| Variant | Description | Desired Vel (m/s) | Min Gap s₀ (m) | Headway T (s) | Aggress. Factor | Special Features |
|---|---|---|---|---|---|---|
| IDM Baseline     | Standard  | 30 | 2.0 | 1.5 | 0.5 | Balanced behavior |
| IDM Conservative | Cautious  | 25 | 3.0 | 2.0 | 0.2 | Safety factor = 1.5 |
| IDM Aggressive   | Dynamic   | 35 | 1.5 | 1.0 | 0.8 | Safety factor = 0.9 |
| IDM Comfort      | Smooth    | 28 | 2.5 | 1.8 | 0.3 | Max jerk = 2.0 |
| IDM Highway      | High-speed| 40 | 3.0 | 1.2 | 0.6 | Perception = 100m |
| IDM City         | Urban     | 15 | 2.0 | 1.5 | 0.4 | Perception = 30m |
| IDM Truck        | Heavy     | 25 | 4.0 | 2.0 | 0.3 | Length = 8.0m |
| IDM Emergency    | Urgent    | 40 | 1.5 | 0.8 | 0.9 | Max accel = 4.0 |
| IDM Adaptive     | Balanced  | 30 | 2.5 | 1.5 | 0.5 | Reaction = 0.2s |
| IDM Defensive    | Safety    | 25 | 4.0 | 2.5 | 0.1 | TTC = 3.0s |

---

## Table A6: Key Configuration Parameters Comparison (p.20)

All planners use dt = 0.1s time step and wheelbase = 2.8m.

| Parameter | Baseline | Aggressive | Conservative | Smooth | Lane Keeper |
|---|---|---|---|---|---|
| *Frenet Planner Weights* | | | | | |
| Lateral (w_l)      | 10.0 | 5.0  | 50.0 | 20.0 | 100.0 |
| Velocity (w_v)     | 1.0  | 0.5  | 1.0  | 2.0  | 1.0 |
| Acceleration (w_a) | 1.0  | 1.0  | 3.0  | 5.0  | 1.0 |
| Progress (w_p)     | 1.0  | 2.0  | 1.0  | 1.0  | 1.0 |
| Jerk (w_j)         | 0.5  | 0.5  | 1.5  | 3.0  | 0.5 |
| *IDM Parameters* | | | | | |
| Desired vel (v₀)    | 30.0 | 35.0 | 25.0 | 28.0 | – |
| Min spacing (s₀)    | 2.0  | 1.5  | 3.0  | 2.5  | – |
| Time headway (T)    | 1.5  | 1.0  | 2.0  | 1.8  | – |
| Max accel (a)       | 2.0  | 3.0  | 1.5  | 1.5  | – |
| Comfort decel (b)   | 3.0  | 4.0  | 2.0  | 2.0  | – |
