# SPACeR Eval Plan (M6)

Evaluation of trained π_θ checkpoints. Companion to [plan.md](plan.md) and
[spacer/STAGE_PLAN.md](spacer/STAGE_PLAN.md).

Decisions locked (from interview, 2026-05-20):

- **Architecture:** two containers, file-based handoff (Option 2).
- **Scope:** phased — quick internal first; full WOSAC only if quick results
  justify the cost.
- **Dataset:** WOMD shards not yet on disk — Phase B includes the download plan.

## Metric coverage — paper vs. ours

Every metric the paper logs (Tables 1, 2, A1, A2 + Fig A1), and where each
stands in our pipeline.

| # | Metric | Paper location | We compute it? | How / when |
|---|---|---|---|---|
| 1 | **Composite** ↑ | Table 1, 2, A1, A2 | ❌ no | WOSAC library → **Phase D** |
| 2 | **Kinematic** ↑ | Table 1, 2, A1 | ❌ no | WOSAC library → **Phase D** |
| 3 | **Interactive** ↑ | Table 1, 2, A1 | ❌ no | WOSAC library → **Phase D** |
| 4 | **Map** ↑ | Table 1, 2, A1 | ❌ no | WOSAC library → **Phase D** |
| 5 | **minADE** ↓ | Table 1, 2, A1, A2 | ⚠️ partial | `eval_quick.py` — needs sentinel-mask patch |
| 6 | **Collision** ↓ | Table 1 | ✅ yes | `eval_quick.py` (GPUDrive `Info`) |
| 7 | **Off-road** ↓ | Table 1 | ✅ yes | `eval_quick.py` — real only after the coord-frame fix |
| 8 | **Throughput** ↑ | Table 1 | ✅ yes | `eval_quick.py` (scenarios/sec) |
| 9 | **D_KL** | Fig A1 | ✅ yes | training loop + `eval_quick.py` |
| 10 | **Log-Likelihood** (= r_h = −NLL) | Fig A1 | ✅ yes | training loop + `eval_quick.py` |
| 11 | **Entropy** | Fig A1 | ✅ yes | training loop + `eval_quick.py` |

**Coverage: 7 of 11.** Split by purpose:

- **Eval-table metrics** (define Tables 1/2/A1/A2): Composite, Kinematic,
  Interactive, Map, minADE, Collision, Off-road, Throughput — we have **4 of
  8** (Collision, Off-road, minADE, Throughput). The missing 4 are the
  WOSAC-composite family; all require the official `waymo-open-dataset`
  library → **Phase D**.
- **Training-dynamics metrics** (Fig A1, in no eval table): D_KL,
  Log-Likelihood, Entropy — we have **all 3**. They show the *mechanism*
  works, not how *good* the result is.

**The single headline metric is Composite** — only obtainable via Phase D
(WOSAC container + WOMD download). Everything else is either a component we
already compute or a training-dynamics signal.

**Ablation scenario:** we run **Variant 4 "KL + r_inf"** (Table A2) — the
best-composite (0.74) row with the fewest tunables (`α=0` no LLH, `w_goal=0`
no goal reward; only `β`). "Best outcome, least effort." See
[plan.md](plan.md#loss-design-variant-4-kl--r_inf--paper-validated-default).

### Paper tables NOT in scope (A3, A4, A5, A6)

| Table | Content | Why excluded |
|---|---|---|
| A3 | PPO training hyperparameters | Not metrics — these are *config*, and we **do** honor them (seed, batch, lr, γ, λ, clip, etc. — our PPO setup). Nothing to "evaluate." |
| A4 | 10 Frenet-based planner variants | Config tables for **classical, non-learned planners** — not metrics, not the SPACeR method. |
| A5 | 10 IDM-based planner variants | Same — Intelligent Driver Model car-following controllers, a parallel rule-based baseline track. |
| A6 | Frenet + IDM parameter comparison | Hyperparameters for the A4/A5 planners. |

**A4/A5/A6 are deliberately out of scope.** They describe rule-based planner
baselines (Frenet lattice planners, IDM controllers) the paper uses as a
*separate comparator track* — they are **not part of the SPACeR method**
(self-play + KL anchoring) and contain **no metrics**. Reproducing them would
mean implementing ~20 classical-planner variants (lattice sampling, IDM
dynamics, all the weight/gain tuning), an entirely separate codebase from the
GPUDrive/CAT-K/SPACeR stack, with zero bearing on validating the SPACeR result
(Variant 4 → Composite 0.74). A full SPACeR-vs-Frenet-vs-IDM comparison would
be a large standalone workstream; it is **not needed** for the reproduction
and is not planned.

## Architecture

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  catk-spacer (existing)  │         │  wosac-eval  (NEW)       │
│  PyTorch + GPUDrive      │         │  TF 2.x + waymo-open-    │
│                          │         │  dataset                 │
│  • Load .pt checkpoint   │         │  • Read rollouts         │
│  • π_θ closed-loop in    │   →→→   │  • Read WOMD GT .tfrec   │
│    GPUDrive on val scenes│  files  │  • Compute composite,    │
│  • Dump rollouts to      │         │    minADE, miss-rate…    │
│    /spacer/eval_runs/    │         │  • Emit JSON metrics     │
└──────────────────────────┘         └──────────────────────────┘
        host:/media/skr/.../SPACeR/spacer  (shared bind mount)
```

The two containers never run code in each other's environment; they
communicate only through files in the shared bind mount.

## Phase A — Quick internal eval (no new deps)

Stays inside `spacer-dev`. Gives us a working number this week even before
the heavy WOSAC bring-up.

| Deliverable | Notes |
|---|---|
| `spacer/eval_quick.py` | Load `.pt` ckpt, closed-loop π_θ rollouts on held-out scenes (e.g. 100 scenes × 1 rollout), compute: collision rate / off-road rate / minADE-vs-logged-GT |
| `spacer/eval_runs/<ckpt_tag>/quick_metrics.json` | Per-ckpt summary; comparable across checkpoints (β sweep, anchor sweep, etc.) |
| Smoke first | 5-scene smoke pass before any full-validation run |

**Gate:** numbers run, are finite, sensible vs random-init baseline.
~hours wall time. Tells us "is training improving anything" without
TF/WOMD.

## Phase B — WOMD `.tfrecord` download (parallel to A)

Official WOSAC needs the **`scenario.proto`-formatted WOMD shards**, not
GPUDrive's processed pickles. Disk budget:

| Split | Approx size | Why needed |
|---|---|---|
| `validation_interactive` | ~150 GB | the WOSAC val split (paper uses this) |
| `testing_interactive` | ~150 GB | optional, for leaderboard submission |
| `training_*` shards | — | not needed (we use GPUDrive's processed form) |

Requires Google Cloud account + accepting Waymo TOS at
[waymo.com/open](https://waymo.com/open). Canonical fetch:

```
gsutil cp -r \
  gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/validation_interactive/ \
  /path/with/200GB+/free/
```

**Plan a host path with ≥ 200 GB free** (250 GB if both interactive splits).
The download is human-in-the-loop (Waymo TOS sign-in); I cannot accept the
TOS on your behalf. Will produce `spacer/WOMD_DOWNLOAD.md` as the checklist.

## Phase C — `wosac-eval` Docker image

Once Phase A justifies committing to full WOSAC.

| Component | Choice |
|---|---|
| Base image | `tensorflow/tensorflow:2.12.0-gpu` (last well-supported TF for `waymo-open-dataset-tf-2-12-0`) |
| Key pip | `waymo-open-dataset-tf-2-12-0`, `protobuf<4`, `numpy<2`, `tensorflow_graphics` |
| Bind mounts (matching catk-spacer layout) | `/spacer:rw`, `/womd:ro` (Phase B shards), `/ckpt:ro` |
| Container name (durable, no `--rm`) | `wosac-eval` |

### Rollout handoff format — the contract between containers

```
/spacer/eval_runs/<ckpt_tag>/rollouts/
  scenario_<id>.npz              # one per WOMD scenario
    keys:
      object_ids   (N,)          # WOMD object IDs (for GT alignment)
      traj         (N, 80, 7)    # 80 timesteps @ 10 Hz; (x, y, z, vx, vy, heading, valid)
      n_rollouts   ()            # = 1 for ADE / 32 for full WOSAC joint metric
```

Rich enough to feed any WOSAC sub-metric without re-running the policy.
`.npz` chosen over `.parquet`/`.tfrecord` for simplicity and zero new deps
in `catk-spacer`. The eval container is free to convert when scoring.

## Phase D — Full WOSAC eval run

Once C is built:

1. **Stage 1 in `spacer-dev`:** `eval_rollout_dump.py` writes `.npz` per
   scenario into `/spacer/eval_runs/<ckpt_tag>/rollouts/`.
2. **Stage 2 in `wosac-eval`:** `wosac_metrics.py` reads rollouts + WOMD GT,
   calls Waymo's metric library, emits `final_metrics.json` with composite +
   sub-scores.
3. **Compare** to Table A2 row 4 (`KL + r_inf`, composite **0.74**).

## Files this plan will create

| Phase | Files |
|---|---|
| A | `spacer/eval_quick.py`, first `quick_metrics.json` |
| B | `spacer/WOMD_DOWNLOAD.md` (checklist for the human-in-the-loop fetch) |
| C | `spacer/wosac_eval/Dockerfile`, `spacer/wosac_eval/requirements.txt`, `spacer/wosac_eval/wosac_metrics.py`, `spacer/eval_rollout_dump.py` |
| D | `spacer/eval_runs/<ckpt_tag>/{rollouts/*.npz, final_metrics.json}` |

## Gating

- **A → B**: Phase A numbers must be sensible (collision rate ↓ vs
  random-init, etc.) before we commit to download cost.
- **B → C**: WOMD shards downloaded and verified before building wosac-eval
  image.
- **C → D**: image builds clean and a 1-scenario smoke runs before full eval.

## Open questions to keep flagged

- **3060 feasibility of WOSAC composite.** The official metric library
  expects batched 32-rollouts-per-scenario evaluation. On 12 GB this likely
  means sequential per-scenario processing. May be slow but feasible.
- **WOMD scenario alignment.** Our training uses GPUDrive's processed
  `validation` (already on disk); the WOSAC eval split is
  `validation_interactive`. These overlap but are not identical — need to
  verify scenario IDs map cleanly between the two formats during rollout
  dump.
- **2 Hz cadence vs WOSAC's 10 Hz requirement.** Our token decoder already
  interpolates to 10 Hz; whether the resulting trajectory shape passes
  WOSAC's kinematic-feasibility check is an open question — only answerable
  by running Phase D.

## Status

- Phase A: ⏳ next
- Phase B: ⏳ parallel to A
- Phase C: ⏳ gated on A pass
- Phase D: ⏳ gated on C bring-up
