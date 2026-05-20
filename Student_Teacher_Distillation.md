# Student–Teacher Distillation for a 5 Hz Reference Model

Summary of the design review. Goal: obtain a **5 Hz π_ref** from the local
**2 Hz** checkpoints (`checkpoints/clsft_E9.ckpt` = CAT-K, `pre_bc_E31.ckpt`
= SMART BC) without the full from-scratch retrain the SPACeR paper used (A.4).

---

## 1. The cadence constraint is structural (verified in code)

`shift = 5` is hardcoded in 3 independent places — not a runtime knob:

- `token_processor.py:44` — `self.shift = 5` → token = 5×0.1 s = **0.5 s** → 2 Hz.
- `agent_decoder.py:60` + `:660` — rollout loop + `pred_traj_10hz[:, t*5:(t+1)*5]`.
- `traj_clustering.py:70` — vocab built at `shift=5`, `# ! don't change`.

Vocab `agent_vocab_555_s2.pkl` = `[2048, 6, 4, 2]` (2048 tokens × 6 intra-token
points 0.0→0.5 s × 4 bbox corners × xy). Filename: **`s2` = seed 2**,
**`555` = tol [0.05,0.05,0.05]** — *not* the shift.

The transformer never sees seconds: temporal signal = token-index delta
(`agent_decoder.py:263`, Fourier) + geometric displacement between token poses.
So "1 step = 0.5 s" only because weights were trained on the 0.5 s vocab +
displacement distribution. → **Checkpoint cannot be re-cadenced by config;
5 Hz = retraining.**

---

## 2. Distillation review

**Logit/KD distillation is ill-posed**: teachers emit a categorical only every
0.5 s; a 5 Hz student needs targets every 0.2 s. No sub-token logit exists.

**Only well-posed form = behavior-level distillation:**
roll teacher out closed-loop → dense 10 Hz trajectory (decoder already
reconstructs `pred_traj_10hz`) → **re-tokenize at shift=2** with a new vocab →
BC + closed-loop SFT the student on those trajectories. This = CAT-K's own
pipeline fed synthetic expert data at shift=2.

"Distilling clsft_E9" ≈ "re-running CAT-K at shift=2." clsft_E9's real value is
**(a) weight initializer, (b) denoised data augmenter / soft target** — not as
the sole data source.

**Key de-risking fact (SPACeR Fig 2):** realism stays ~0.73 even with a
0.3M-param / 0.636-realism reference. π_ref is a *soft prior, not an imitation
target* → the 5 Hz student need not approach clsft_E9 quality; "good enough +
high entropy" suffices.

---

## 3. Multi-teacher: orthogonal to Hz, but still worth it

**Does NOT fix Hz.** A 2nd 2 Hz teacher adds another 0.5 s opinion, zero
sub-0.5 s resolution. Both checkpoints decide on the same phase (no stagger).
Temporal resolution comes from **dense 10 Hz reconstruction + shift=2
re-tokenization** — one teacher already provides it.

**DOES fix the failure mode the Hz fix creates.** Single-teacher distillation
from clsft_E9 = BC-on-BC → entropy/mode collapse → exactly what the SPACeR
α-ablation says wrecks the log-likelihood reward + KL. Two teachers counter it:

- `clsft_E9` (CAT-K) → closed-loop-stable, denoised, on-road. **Behavior source.**
- `pre_bc_E31` (SMART BC) → higher entropy, multimodal. **Diversity source.**

**Combine correctly — mixture-of-rollouts, NOT logit-averaging:**
per scene, sample which teacher rolls out (weight **~80/20 toward clsft**;
`pre_bc` is open-loop and *drifts* closed-loop, so 50/50 injects unrealistic
off-road data). Re-tokenize all rollouts at shift=2, BC student on the pool →
target becomes empirical mixture (higher entropy, multimodal).
Probability-averaging the two categoricals is unsafe (mode interpolation:
left+right → straight-into-obstacle). Alternative: `pre_bc` as label-smoothing
prior only, `clsft` provides the trajectory.

Caveats: doesn't reduce S2.5 cost (+1 rollout pass); per Fig 2 it's cheap
insurance against entropy collapse, not a quality lever — keep it small.

---

## 4. Recommended plan (tiered, evidence-gated)

| Tier | Trigger | Action |
|------|---------|--------|
| **0** | First, always | Run M2–M4 with stock `clsft_E9` at 2 Hz; make π_θ also 2 Hz. Eq. 5 stays exact when both share cadence/vocab. Cadence is a *reactivity* problem, not a KL-validity problem. Zero training. |
| **1** | Only if M4 reactivity fails | **S2.5**: regenerate K-disk vocab at **shift=2, K≈200** (not 2048); **warm-start backbone from clsft_E9**, re-init token-embed + `token_predict_head`; closed-loop SFT on **WOMD + multi-teacher (clsft 80 / pre_bc 20) rollout mix**. |
| **2** | If Tier-1 student underperforms | Full from-scratch BC+CLSFT at shift=2 on WOMD (paper's literal A.4 path; expensive). |

**Hard coupling:** Eq. 5 requires π_θ and π_ref over the *identical*
vocab+cadence. Define the 5 Hz K-disk vocab **once, freeze it, share it**
between distilled π_ref and the RL action head — single source of truth.

**Implementation gotchas (Tier 1):**
1. `tol_dist=0.05` was tuned for 0.5 s endpoints; at 0.2 s shrink it ~2.5×
   (`Kdisk_cluster` always pads to N with near-duplicates if coverage exhausts).
2. shift=5 hardcode sites: `token_processor.py:44`, `agent_decoder.py:60`,
   the `*5`/`//5` reconstruction, `time_span/shift`, `n_step_token` 18→45.
3. **Promotion gate** before student serves as π_ref: (a) WOSAC composite within
   tolerance at 5 Hz, (b) **entropy ≥ teacher** (peaky π_ref collapses π_θ).
   Chase entropy + "good enough", not teacher parity.

**Bottom line:** distillation is sound *only* as behavior-level (not logit KD),
best framed as "warm-started CAT-K re-training at shift=2 with clsft_E9 as
init + denoised augmenter." Multi-teacher is orthogonal to Hz but is the
defense against BC-on-BC entropy collapse: `clsft` for behavior, `pre_bc` for
entropy, via mixture-of-rollouts. Defer all of it behind the M4 reactivity gate.
