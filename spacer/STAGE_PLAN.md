# SPACeR — Modular Stage/Gate Plan

Each stage = one deliverable + one **gate** (an automated `spacer/test_*.py`
giving a binary PASS/FAIL on a quantitative threshold). A stage is "done" only
when its gate PASSes; every gate result is recorded in `test.md`. Each stage
also defines an explicit **on-fail** action so a failure is a decision point,
not a dead end.

Gate principles:
- **Measurable**: numeric threshold, not "looks right".
- **Isolated**: each gate tests *only* its stage (reuse proven upstream code).
- **Cheap first**: the decisive correctness check before the expensive one.
- **Recorded**: PASS/FAIL + numbers appended to `test.md`.

---

## ⚙️ Loss invariant: Variant 4 (KL + r_inf) is the default

**Paper Table A2** ablation conclusion: KL (Eq. 5) is the only load-bearing
anchoring term; LLH (Eq. 3, α-reward) and goal-reward are both droppable
without composite loss. We adopt **Variant 4 — "KL + r_inf"** as the default:

```
Loss = − L_PPO( r_inf )  +  β · KL(π_θ ‖ π_ref)
  r_inf = −0.75·𝟙[collision] − 0.75·𝟙[off-road]   (no goal channel)
  α = 0,  w_goal = 0
```

Wired in `train_spacer.py` (`build_env` EnvConfig overrides + `run(alpha=0.0)`
default). M5c/M5e sweeps run **under this reduced loss**; β is the only scalar
to tune. Full variant table & rationale: `../Architecture.md`.

---

## S0 — Environment & components  ✅ PASS

| | |
|---|---|
| Goal | GPUDrive engine + CAT-K deps coexist on RTX 3060; π_ref + sim work |
| Paper | infra (Sec 3.2 reference model; GPUDrive self-play env) |
| Gate | Tests 1–2: strict ckpt load (0 missing/unexpected, 2048-head); policy rollout runs |
| **PASS** | met — `catk-spacer:latest`, Test 1 & 2 green |

## S1 — GPUDrive→SMART adapter  ✅ PASS

| | |
|---|---|
| Goal | π_ref scores real GPUDrive scene state |
| Paper | prerequisite for Eq. 3 & Eq. 5 (paper-required, unreleased) |
| Deliverable | `gpudrive_to_smart.py` (reuses CAT-K `get_agent_features`/`preprocess_map`) |
| Gate | T3 runs → real `(A,16,2048)` logits; T4 GT-token NLL ≪ random; T5 live rollout |
| **PASS** | met — T4 NLL 3.46 vs 7.62 (top-1 ≈900× chance); T5 live works |
| Note | T5: token-NLL (Eq. 3) weakly discriminative — *expected*; real signal is KL → validated at S4 |

## S2 / M1 — token→state driver  ✅ PASS

| | |
|---|---|
| Goal | decode any token → global pose → drive sim faithfully |
| Paper | Sec 4.1 tokenized trajectory action space |
| Deliverable | `token_decode.py` (reuses CAT-K `transform_to_global`) |
| Gate | (a) decoder fed `gt_idx` == tokenizer `gt_pos/gt_head`; (b) `state` action places agent at commanded pose |
| **PASS** | met — (a) **0.00** err / 1048 steps; (b) pos err 0.0 m, yaw 3.6e-7 |

---

## ⚖️ Cadence invariant (applies to S3–S6 and S2.5)

**Eq. 5's closed-form KL requires π_θ and π_ref over the *same* discrete action
space at the *same* cadence.** A 5 Hz student + 2 Hz reference → action spaces
don't align → the closed-form KL (SPACeR's core efficiency trick) **breaks**.
So whatever cadence is chosen, **both must be at it**. Consequence: you cannot
"just make the policy faster" — changing cadence means producing a *new
reference at that cadence too* (hence S2.5 is a *reference* stage, not a policy
tweak). Default path runs both at the checkpoint-native **0.5 s / 2 Hz**.

## S2.5 — OPTIONAL: 5 Hz reference distillation  ⏳ (conditional on S5/M4)

> **Run only if** S5/M4 shows the 2 Hz cadence is too sluggish (see M4
> reactivity diagnostic). Otherwise **skipped** — the default pipeline stays at
> 2 Hz, zero extra training.

| | |
|---|---|
| Goal | a **5 Hz** tokenized reference distilled from `clsft_E9` (teacher), so π_θ *and* π_ref can both run at 5 Hz — recovers A.4's responsiveness design |
| Why needed | A.4 used 5 Hz *"to make the self-play policy more responsive"*; our public ckpt is locked at 0.5 s/2 Hz. Per the cadence invariant, getting 5 Hz means a **new 5 Hz reference**, not a faster policy alone. Distillation from the strong frozen teacher is far cheaper than A.4's full BC+CLSFT on raw human data |
| Deliverable | `distill_5hz.py`: (1) re-cluster trajectories at **0.2 s** → new ~2048-token 5 Hz vocab (reuse CAT-K `traj_clustering.py`); (2) roll out `clsft_E9` (teacher) on WOMD/GPUDrive scenes for dense behavior; (3) train a small SMART/MLP **5 Hz student** to match teacher trajectory distribution; freeze as `pi_ref_5hz` |
| Gate test | `test_s25_distill.py` |
| **PASS criteria** (all) | 1. student reproduces teacher behavior at 0.2 s: rollout ADE vs teacher below a set threshold (e.g. < 1 m mean) & token-NLL ≪ random; 2. **discriminative KL retained at 5 Hz**: re-run S4 Part-B monotone contrast at the new cadence — `KL(random) − KL(good) ≥ 0.5 nats`; 3. both π_θ-side decode and π_ref share the *same* 5 Hz vocab (cadence invariant holds) |
| On fail | distillation insufficient → larger student / more teacher data; if still weak → **document limitation, revert to 2 Hz** (accept coarser control as a Strategy-A consequence) |

## S2.6 — OPTIONAL: token-cluster marginalization ("S2.5-lite")  ⏳ (alt/parallel to S2.5)

> Orthogonal to S2.5 — addresses the **vocab-compute tax** (head/KL/exploration
> at 2048), **not** cadence. Use *in addition to* S2.5 (cadence) or instead, if
> the issue proves vocab-driven and not cadence-driven.

| | |
|---|---|
| Goal | shrink the *effective* action space from 2048 → K (e.g. 200) by **clustering** `agent_token_all` and **marginalising** π_ref's frozen output to the K clusters — **no retraining of π_ref** |
| Why needed | paper Table R4 / our own M5 finding: vocab 2048 inflates π_θ head (304k vs paper ~65k) + KL-sum memory + RL exploration; reduces convergence speed on the 3060 |
| Mechanism (no retraining) | (1) offline k-means on the 2048 token contour templates → K groups; (2) at runtime `p_ref_cluster(g) = Σ_{a∈g} π_ref(a)` — *exact*, just summing frozen probabilities; (3) π_θ head = K; (4) sim/decode via per-cluster representative token |
| Deliverable | `cluster_tokens.py` (offline k-means) + `marginalize_ref()` in `anchor.py` + thin TokenPolicy variant with K-head |
| Gate test | `test_s26_marginalize.py` |
| **PASS criteria** | 1. marginalised probabilities sum to 1 (numerical sanity); 2. KL(π_θ_K ‖ p_ref_K) ≥ 0 and **monotone discrimination retained** (mirror S4 Part-B at K); 3. M5b-style faithful loop still trains (KL ↓ under β>0); 4. head size + Σ memory measurably reduced (e.g. K=200 ⇒ head ≈ 65k, KL Σ over 200) |
| Caveats (honest) | **Data-processing inequality**: KL over coarser partition ≤ KL over 2048 ⇒ anchoring signal is *provably weaker* than at 2048 (within-cluster divergence becomes invisible). Lossy post-hoc coarsening of a 2048 space ≠ the paper's *natively-trained* 200 reference. Choice of cluster representative for sim-decode is a design knob. |
| On fail | KL signal too weak after coarsening → either pick larger K (e.g. 400/800) or fall back to 2048; if compute still infeasible at any K → S2.5 (full distillation) |

---

## S3 / M2 — π_θ 2048-token policy head  ▶ NEXT

| | |
|---|---|
| Goal | decentralized policy emitting a categorical over the 2048 agent tokens |
| Paper | the policy **π_θ** (decentralized, tokenized action) |
| Deliverable | `policy_token.py` — GPUDrive late-fusion MLP backbone, head swapped 91→2048 (per agent-type vocab); consumes GPUDrive local obs |
| Gate test | `test_m2_policy.py`: random-init forward on a real GPUDrive_mini obs batch |
| **PASS criteria** (all) | 1. output shape `[n_controlled, 2048]`; 2. valid distribution: `softmax` sums to 1±1e-4, **no NaN/Inf**; 3. init entropy ≈ ln 2048 = 7.62 (within ±0.3 — i.e. ~uninformative at init, no degenerate collapse); 4. sampled tokens feed `token_decode` → **finite** poses for all agents; 5. param count logged (backbone ≈ baseline + 2048·hidden head) |
| On fail | wrong obs dim → fix obs adapter; collapsed entropy → head init; non-finite → dtype/masking |

## S4 / M3 — Eq. 3 + Eq. 5  ⏳  **(research gate — make/break)**

| | |
|---|---|
| Goal | likelihood reward + closed-form KL between π_θ and π_ref, aligned in time |
| Paper | **SPACeR's central contribution** — Eq. 3 `r_h=log π_ref(a_t|s_t)`, Eq. 5 `D_KL=Σ_{2048} π_θ log(π_θ/π_ref)` |
| Deliverable | `anchor.py` — cadence alignment (π_θ ↔ π_ref at the checkpoint's native **0.5 s / 2 Hz** token-steps; paper's 5 Hz n/a with fixed public ckpt), `kl()`, `r_humanlike()` |
| Gate test | `test_m3_anchor.py`, two parts: |
| **Part A — mechanical** (all must hold) | KL ≥ 0 everywhere; KL(π_ref‖π_ref) ≤ 1e-5; `r_h` finite & equals `log π_ref` at executed token; π_θ↔π_ref step indices aligned (no off-by-one vs `tokenize_agent` 5-step window) |
| **Part B — signal validation** (the real gate) | Build π_θ proxies and check **monotone discrimination**: `KL(ref‖ref)≈0 < KL(good-policy rollout) < KL(random rollout) < KL(uniform)`; and `r_h(good) − r_h(random) ≥ 0.5 nats` |
| **PASS criteria** | Part A all hold **AND** Part B monotone with `KL(random) − KL(good) ≥ 0.5` (ideally ≥ 1.0). This is the signal Test 5 could *not* show via token-NLL — it must appear via KL. |
| On fail | (i) margin small → refine adapter fidelity (map type-enum, traffic-light) and retest; (ii) still weak → documented finding: KL signal limited on this stack → fall back to Strategy B (own tiny aligned reference) or accept reduced anchoring. Not a silent pass. |

## S5 / M4 — PPO + KL loop (Eq. 2/1)  ⏳

| | |
|---|---|
| Goal | the SPACeR training algorithm, scaled-down on the 3060 |
| Paper | Eq. 2 `L = L_PPO − β·D_KL`; Eq. 1 `r = r_task + α·r_humanlike` |
| Deliverable | `train_spacer.py` — adapt GPUDrive PufferLib PPO; π_θ token policy; per-rollout π_ref scoring via adapter; loss assembly |
| Gate test | `test_m4_train.py`: short scaled run (e.g. ≤8 worlds, ≤2–5 M steps) + a β=0 ablation |
| **PASS criteria** (all) | 1. **stability**: completes, no OOM on 12 GB, no NaN/divergence; throughput (steps/s, VRAM) logged; 2. **learning**: task reward trends up vs random-init over the short run (not noise); 3. **KL bounded**: D_KL stays finite and does not collapse to 0 nor explode; entropy not prematurely collapsed; 4. **anchoring effect**: β>0 run has lower mean KL / higher `r_h` / better realism-proxy than the β=0 ablation (measurable, directionally per paper Table R3) |
| **Reactivity diagnostic** (decides S2.5) | Log **collision rate** and **off-road rate** of the trained policy. If, *despite* stable training and a working anchoring effect, **collision/off-road stays high and behaviour is visibly sluggish** (slow reaction at the 0.5 s decision boundary), that is the signature of the **2 Hz cadence being too coarse** → **trigger optional S2.5 (5 Hz reference distillation)**. If reactivity is acceptable, S2.5 is skipped. |
| Event-detection sub-check | ✅ **resolved (Test 11)** — `state`-dynamics does NOT bypass collision/off-road event detection; events fire correctly. The original M5c `r_task=0` was an *edge-triggered* artifact of `collision_behavior="ignore"`. Fix: `train_spacer.build_env` now sets `collision_behavior="stop"` (level-triggered, sustained penalty ⇒ clear gradient). So the S2.5 trigger is **not** about event detection; the cadence-sluggishness question itself is still open, but separate, and tested only by a longer run on the fixed config. |
| On fail | reward scaling/α,β → tune; grad/NaN → clip/precision; throughput infeasible → reduce worlds, defer to S5; **persistent high collision + sluggishness despite `stop` → not a tuning bug, it's the 2 Hz cadence → S2.5** |

## S6 / M5 — scale tuning + scaled reproduction  ⏳

| | |
|---|---|
| Goal | maximize useful training within 12 GB; longer scaled run |
| Paper | reproduction (paper: 1B steps / A100 — out of reach; target *scaled-down*) |
| Deliverable | tuned config (worlds/batch/steps); longer run; metrics log |
| Gate test | `test_m5_scale.py` / run logs: sustained run + realism-proxy trend |
| **PASS criteria** | 1. stable long run within 12 GB (no OOM, no drift); 2. **realism proxy improves over training** (mean KL↓ and/or token-NLL of rollout↓, or a mini composite proxy ↑) — i.e. the policy becomes more human-like; 3. throughput + final proxy documented, with explicit gap-vs-paper stated |
| Optional final | true WOSAC composite in a *separate* TF/waymo env (off critical path) for paper-comparable number |
| On fail | not "fail" but ceiling: document achieved scale/quality vs paper honestly |

## M5d — multi-world rollout support  ⏳ (operational unblocker)

| | |
|---|---|
| Goal | lift the world-0-only constraint in `rollout`/adapter so π_θ acts across many parallel envs per iter (the M4-smoke `--scenes=2` failure showed `tok` vs `cmask[0]` mismatch — currently we cycle scenes via `dataset_size>1, batch_size=1`, i.e. **one world at a time**) |
| Why needed | throughput on the 3060: single-world rollout caps at ~0.75 it/s ⇒ longer runs are slow. Multi-world batches the rollout + π_ref forward, getting closer to the GPU's actual capacity |
| Deliverable | rollout/score_ref refactor for `num_worlds > 1`: per-world buffers, per-world adapter→HeteroData batched (or `Batch.from_data_list` of N scenes), per-world theta_ids; correctness regression of M5b/M5c at `num_worlds > 1` |
| Gate test | `test_m5d_multiworld.py` |
| **PASS criteria** | 1. M5b loop runs identical (within noise) for `num_worlds=1` vs `num_worlds=N` on the same seed; 2. throughput it/s strictly improves with N; 3. no OOM at the chosen N on 12 GB |
| On fail | leave at single-world; document throughput ceiling |

## M5e — β-only sweep  ⏳ (extends M5c's binary ablation)

> **Scope tightened by Variant 4 decision** (Table A2): α is now fixed at 0
> (LLH dead-weight) and w_goal=0 (goals unnecessary). The only anchoring
> scalar left to tune is β. This collapses the original α×β grid to a 1-D
> sweep — cheaper, cleaner.

| | |
|---|---|
| Goal | find β optimum **for our 2048-vocab reference** (paper's `β=0.1` tuned on its 200-vocab — KL magnitude scales with vocab ⇒ optimum likely shifts) |
| Why needed | M5c was binary (`β∈{0, 0.1}`); we should sweep β∈{0, 0.01, 0.1, 1.0} (paper Table R3) under Variant 4 (α=0, w_goal=0) |
| Deliverable | `sweep.py` driver running `train_spacer.py --alpha 0 --beta β` for each β in the grid (sequential separate processes — see M5c CUDA-heap finding), bounded iter budget; CSV/JSON log |
| Gate test | `test_m5e_sweep.py` (or just a logged run) |
| **PASS criteria** | 1. all runs complete (no NaN/OOM); 2. per-β KL / r_task (collision+off-road) trends recorded; 3. identify best β by reactivity readout (collision/off-road) + final KL stability |
| Optional follow-up | if reactivity readout suggests goal-anchoring would help, run one Variant 3 (KL + Goal) reference cell |
| On fail | grid too expensive on 3060 ⇒ shrink grid to {0, 0.01, 0.1}; document |

---

## Status snapshot

```
S0 ✅  S1 ✅  S2/M1 ✅  │  S3/M2 ✅  S4/M3 ✅  S5/M4 ✅  M5a-c ✅
        proven foundation │  mechanism implemented & validated end-to-end

Remaining (NOT mechanism — scale/measurement/options):
  · Reactivity Info-tensor readout  → decides S2.5
  · M5d  multi-world rollout                  (throughput)
  · M5e  α/β multi-point sweep                (re-tune for 2048-vocab)
  · S2.6 token-cluster marginalization        (S2.5-lite, vocab-compute, no retrain)
  · S2.5 5 Hz reference distillation          (cadence; only if reactivity demands)
  · S6/M5 long convergent run + WOSAC eval    (paper-grade — 3060 ceiling)
```

**Critical gate is S4/M3**: everything before it makes Eq. 3/5 *computable*;
S4 Part B is where SPACeR's core mechanism (KL anchoring) is *validated on this
stack* — a genuine research checkpoint, not just code that runs.
