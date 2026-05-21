# SPACeR Training Configuration — Paper vs. Ours

Consolidated reference for designing training runs. All paper figures
extracted from `SPACeR.pdf` (arXiv:2510.18060v2, ICLR 2026); Table A3 also
in [tables.md](tables.md). Companion to [plan.md](plan.md).

> Use this doc when sizing/configuring a training run. It records what the
> paper did, what we do, and what the RTX 3060 12 GB can actually support.

---

## 1. Paper's training setup (extracted)

### 1.1 Compute / scale

| Item | Paper value | Source |
|---|---|---|
| GPU | 1× A100 80 GB PCIe | A.3 |
| Total env-steps | **1×10⁹** | A.3 / Table A3 (`total_timesteps`) |
| Training scenarios | **10,000**, resampled from WOMD | A.3 (p.~18) |
| WOMD training split size | ≈500,000 scenarios | A.3 |
| Parallel worlds — no reference model | **600** | A.3 |
| Parallel worlds — reference-model (SPACeR) | **300** | A.3 |
| Unique scenarios per batch (cap) | **200** | A.3 |
| Wall time | ~24–48 h | Sec 4 |

**Scenario replication.** 300 worlds but ≤200 *unique* scenarios per batch ⇒
~100 worlds run **duplicate** scenarios. The cap is on scenario *diversity*,
not batch size — a memory/speed optimisation (loading a distinct scene into
the Madrona sim is expensive). PPO is on-policy; there is **no** replay
buffer / old-new mix — each batch is fresh rollouts, ≤200 distinct scenes.

### 1.2 Episode / simulation

| Item | Paper value | Source |
|---|---|---|
| WOMD scenario length | 9 s | Sec 4 |
| Episode (rollout) length | 8 s closed-loop | Sec 4 |
| Control cadence | 5 Hz | Sec 4 / A.3 |
| Max agents per scenario | up to 128 | A.4 |
| Map elements per agent (ref-model setting) | 200 → **120** (trimmed) | A.3 |

### 1.3 Network

| Item | Paper value | Source |
|---|---|---|
| Input embedding dim | 64 | A.3 |
| Hidden dim | 128 | A.3 |
| Dropout | 0.01 | A.3 |
| BC reference capacity | ≈2× the self-play policy | A.3 |
| BC reference: data / epochs / val-acc | full WOMD / 60 epochs / 92% | A.3 |

### 1.4 PPO hyperparameters (Table A3)

| Param | Value | | Param | Value |
|---|---|---|---|---|
| seed | 42 | | norm_adv | true |
| total_timesteps | 1×10⁹ | | clip_coef | 0.2 |
| batch_size | 131,072 | | clip_vloss | false |
| minibatch_size | 8,192 | | vf_clip_coef | 0.2 |
| learning_rate | 3e-4 | | ent_coef | 1e-4 |
| anneal_lr | false | | vf_coef | 0.3 |
| gamma | 0.99 | | max_grad_norm | 0.5 |
| gae_lambda | 0.95 | | update_epochs | 4 |

### 1.5 SPACeR loss (Variant 4 — our chosen ablation)

| Param | Paper value | Source |
|---|---|---|
| β (KL weight) | **0.01** (tokenised model robust over 0.01–1.0) | A.3 |
| α (LLH weight) | 0 (Variant 4 — LLH dropped) | Table A2 |
| w_goal | 0 (goals "unnecessary") | Table A2 |
| collision / off-road weight | −0.75 / −0.75 (r_inf) | A.2 |
| Reference vocab size | 200 (paper's *own* trained reference) | Sec 3 |
| Eval rollouts per scenario (WOSAC) | 32 | A.4 |

---

## 2. Comparative table — paper vs. ours vs. 3060

**Legend:** ✅ matched & in use · ⚠️ differs · 🔻 scaled down · ❌ not
implemented (compact loop) · ⛔ out of reach on 3060.

### Scale / compute

| Item | Paper | Ours | 3060 status |
|---|---|---|---|
| Hardware | A100 80 GB | RTX 3060 12 GB | — |
| Total env-steps | 1×10⁹ | ~1.4×10⁶ (200 iter) | ⛔ ~700× short — hard ceiling |
| Training scenarios | 10,000 | 1,000 (`GPUDrive_mini`) | 🔻 have 1k; compute-bound not data-bound |
| Parallel worlds | 300 (ref-model) | 32 | ✅ W=32 tested; W=64 expected to fit; true OOM ceiling untested |
| Unique scenarios / batch | 200 (300 worlds, ~100 replicas) | 32 (= worlds; all distinct, no replication) | ✅ tested |
| Control cadence | 5 Hz | 2 Hz (`clsft_E9` native) | ⚠️ 5 Hz needs Stage S2.5 |

### PPO hyperparameters

| Param | Paper | Ours | Status |
|---|---|---|---|
| learning_rate | 3e-4 | 3e-4 | ✅ matched |
| anneal_lr | false | false | ✅ matched |
| max_grad_norm | 0.5 | 1.0 | ⚠️ differs |
| batch_size | 131,072 | ~4k decisions/iter | 🔻 scaled |
| minibatch_size | 8,192 | — | ❌ no minibatching |
| update_epochs | 4 | 1 | ❌ one update/iter |
| gamma / gae_lambda | 0.99 / 0.95 | — | ❌ no GAE / discounting |
| clip_coef | 0.2 | — | ❌ no ratio-clip |
| vf_coef / vf_clip_coef | 0.3 / 0.2 | — | ❌ value head unused in loss |
| ent_coef | 1e-4 | 0 | ❌ no entropy bonus |
| norm_adv | true | — | ❌ no advantage norm |
| seed | 42 | not pinned | ⚠️ |

Our loop: `l_pg = −(logₚ · r̄)`, `loss = l_pg + β·KL` — REINFORCE-style PG +
KL, **not** clipped PPO with GAE. The **KL anchoring (SPACeR contribution)
is exact**; the PPO scaffolding is reduced.

### SPACeR loss design

| Param | Paper | Ours | Status |
|---|---|---|---|
| β (KL weight) | 0.01 | 0.1 | ✅ both tested — Test 12 (0.1), Test 13 (0.01) |
| α (LLH) | 0 | 0 | ✅ matched |
| w_goal | 0 | 0 | ✅ matched |
| collision / off-road weight | −0.75 / −0.75 | −0.75 / −0.75 | ✅ matched |
| Eq. 5 closed-form KL | yes | yes (exact per-agent) | ✅ matched |
| Reference vocab | 200 (own ref) | 2048 (`clsft_E9`) | ⚠️ Strategy-A substitution |

---

## 3. Implications for designing our training runs

**What is faithfully reproduced:** the SPACeR mechanism — Eq. 1/2/3/5, the
closed-form KL anchor, Variant 4 reward design. These are matched.

**Two structural gaps to decide on before a "real" run:**

1. **Scale (hardware-bound, not fixable).** 1×10⁹ env-steps is ~700× our
   budget. Per [plan.md](plan.md) §"Training budget guidance", the knee of
   the paper's curves is ~1×10⁸ steps ≈ 2 days on the 3060; a meaningful
   trend shows by ~2×10⁷ steps ≈ 11 h (~5,000 iters @ W=32). A run is sized
   in **iters**, not scenes — 1,000 scenes is ample.

2. **PPO machinery (a deliberate simplification — fixable).** Our compact
   loop omits GAE, ratio-clipping, value loss, minibatch epochs, advantage
   normalisation, entropy bonus. To match Table A3, `train_spacer.py` would
   need full PPO wired in. Open decision: leave compact (the KL anchor — the
   contribution — is exact regardless) vs. invest in full-PPO fidelity.

**Long-run config — LOCKED** (post Tests 15–18; ported PPO; corrected frame):

| Knob | Value | Rationale |
|---|---|---|
| `--worlds` | **48** | Test 17 — safe ceiling (~1.8 GB headroom); W=64 is a knife-edge, unsafe for a 10 h run |
| `--iters` | **3,500** | ✅ confirmed — ~10 h @ ~10 s/iter; reaches the Fig-A1 knee (~2×10⁷ env-steps; W=48 collects 1.5× more/iter than the W=32 budget table assumed, so 3.5k ≈ 5k @ W=32) |
| `--ckpt-every` | **250** | ✅ confirmed — 14 checkpoints; ~42 min max crash-loss; doubles as the 14-point post-hoc eval/learning curve |
| `--beta` | 0.1 | inside paper's robust band; Test 12/14/16 baseline. (0.01 = paper-canonical; needs paper-scale budget — see Test 13) |
| `--scenes` | **1,000** | ✅ `build_env` split fix done — `--split training` (1000 scenes); `validation/` (150) held out for eval_quick |
| `--split` | **training** | trains on `training/`; eval stays on `validation/` (held out) |
| launch | `docker exec -d` + `nohup` | survives session/terminal drop ([[nohup-training-launches]]) |

Launch (gated only on the 200-iter validation run confirming the port):
`train_spacer.py --mode smoke --iters 3500 --worlds 48 --scenes 1000
--split training --beta 0.1 --ckpt-every 250 --ckpt-dir /spacer/checkpoints`

---

## Status

Extracted 2026-05-21. Update if a longer run, a full-PPO rewrite, or a
cadence change (S2.5) is undertaken.
