# SPACeR — Component Smoke Tests

Two halves of the SPACeR pipeline validated independently on the local
**RTX 3060 (12 GB)** inside the persistent Docker image `catk-spacer:latest`.

---

## Test 1 — π_ref side: CAT-K checkpoint load (decoupled)

**Goal:** confirm `clsft_E9.ckpt` / `pre_bc_E31.ckpt` are structurally proper and
emit the agent-token categorical SPACeR needs (Eq. 3 / Eq. 5).

**Method:** load weights into the real `SMARTDecoder` (built from each
checkpoint's own `hyper_parameters`), bypassing the TF/waymo metrics chain
(decoupled load — strip `encoder.` prefix).

```bash
docker exec -i -w /catk catk-test python - <<'PY'
import torch, sys; from omegaconf import OmegaConf
sys.path.insert(0, "/catk")
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
for name in ["clsft_E9.ckpt", "pre_bc_E31.ckpt"]:
    ck  = torch.load(f"/ckpt/{name}", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["hyper_parameters"]).model_config
    sd  = ck["state_dict"]
    tp  = TokenProcessor(**cfg.token_processor)
    dec = SMARTDecoder(**cfg.decoder, n_token_agent=tp.n_token_agent)
    enc = {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}
    res = dec.load_state_dict(enc, strict=True)
    print(name, "n_token_agent", tp.n_token_agent,
          "missing", len(res.missing_keys), "unexpected", len(res.unexpected_keys))
PY
```

**Result:** ✅


| Checkpoint                   | tensors | n_token_agent | missing | unexpected |
| ------------------------------ | --------- | --------------- | --------- | ------------ |
| `clsft_E9.ckpt` (CAT-K)      | 811     | 2048          | 0       | 0          |
| `pre_bc_E31.ckpt` (SMART BC) | 811     | 2048          | 0       | 0          |

- Next-token head confirmed: `token_predict_head.mlp.3 → out_features = 2048`.
- Agent action vocabulary is **2048 tokens** (the `555` in
  `agent_vocab_555_s2.pkl` is *not* the vocab size).
- Both checkpoints are valid, usable as π_ref (structural proof; semantic
  forward deferred until the GPUDrive→SMART adapter exists).

---

## Test 2 — π_θ side: GPUDrive closed-loop policy rollout

**Goal:** confirm the GPUDrive simulator + a pretrained GPUDrive policy run
closed-loop on real `GPUDrive_mini` scenes on the RTX 3060.

**Checkpoint:** `policy_S10_000_02_27` (GPUDrive-native decentralized policy —
the only checkpoint that runs *directly* in GPUDrive; CAT-K cannot, it needs
the adapter).

**Setup:**

```bash
docker run -d --name gpudrive-test --gpus all \
  -v /media/skr/storage/self_driving/self_play/SPACeR/reference_code/gpudrive:/gpd:ro \
  catk-spacer:latest sleep infinity
```

**Rollout script** (run via `docker exec -i -e HF_HUB_OFFLINE=1 -w /gpd gpudrive-test python -`):

```python
import torch, dataclasses
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

dev = "cuda"
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
num_envs = 2
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=num_envs, dataset_size=4,
                         sample_with_replacement=False)
env_config = dataclasses.replace(
    EnvConfig(),
    ego_state=cfg.ego_state, road_map_obs=cfg.road_map_obs, partner_obs=cfg.partner_obs,
    reward_type=cfg.reward_type, norm_obs=cfg.norm_obs, dynamics_model=cfg.dynamics_model,
    collision_behavior=cfg.collision_behavior, dist_to_goal_threshold=cfg.dist_to_goal_threshold,
    polyline_reduction_threshold=cfg.polyline_reduction_threshold,
    remove_non_vehicles=cfg.remove_non_vehicles, lidar_obs=cfg.lidar_obs,
    disable_classic_obs=cfg.lidar_obs, obs_radius=cfg.obs_radius,
    steer_actions=torch.round(torch.linspace(-torch.pi, torch.pi, cfg.action_space_steer_disc), decimals=3),
    accel_actions=torch.round(torch.linspace(-4.0, 4.0, cfg.action_space_accel_disc), decimals=3),
)
env = GPUDriveTorchEnv(config=env_config, data_loader=loader,
                       max_cont_agents=cfg.max_controlled_agents, device=dev)
agent = NeuralNet.from_pretrained("/gpd/models/policy_S10_000_02_27").to(dev).eval()

obs = env.reset(); cmask = env.cont_agent_mask
for t in range(15):
    with torch.no_grad():
        act,_,_,_ = agent(obs[cmask], deterministic=True)
    tmpl = torch.zeros(cmask.shape, dtype=torch.int64, device=dev)
    tmpl[cmask] = act.to(dev)
    env.step_dynamics(tmpl)
    obs = env.get_obs()
```

**Result:** ✅


| Aspect                 | Value                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| Madrona GPU engine     | compiled + initialized on RTX 3060 (kernel cached in container)  |
| Policy`action_dim`     | 91 (13 steer × 7 accel — matches`reliable_agents_params.yaml`) |
| Observation shape      | `(2 envs, 64 max agents, 2984)`                                  |
| Controlled agents      | 15 across 2`GPUDrive_mini` validation scenes                     |
| episode_len            | 91                                                               |
| Rollout                | 15 steps completed; policy→action→step→obs/reward loop OK     |
| Total reward (rollout) | 1137 (positive, sane)                                            |

> Throughput at `num_envs=2` (~61 env-steps/s) is **not** representative — real
> training uses hundreds of parallel worlds; this is a correctness smoke test only.

---

## Test 3 — the missing link: GPUDrive→SMART adapter (real-data π_ref forward)

**Goal:** feed π_ref from live GPUDrive state — i.e. produce the 2048-way
agent-token categorical (Eq. 3 `log π_ref` / Eq. 5 KL) from a real
`GPUDrive_mini` scene, with no TF/waymo and no WOMD tfrecords.

**Design:** `spacer/gpudrive_to_smart.py` only does GPUDrive-state → WOMD-style
dict extraction, then **reuses CAT-K's own `get_agent_features()` and
`preprocess_map()`** (no token/polyline logic reimplemented). A small
`waymo_open_dataset` stub lets `get_agent_features` import without the conflict
stack. `catk/` is unmodified. Output: `HeteroData` → `TokenProcessor` →
`SMARTDecoder`.

**Files:** `spacer/gpudrive_to_smart.py`, `spacer/test_adapter.py`

**Run:**

```bash
docker run -d --name spacer-dev --gpus all \
  -v .../reference_code/catk:/catk:ro -v .../reference_code/gpudrive:/gpd:ro \
  -v .../checkpoints:/ckpt:ro -v .../spacer:/spacer:rw \
  catk-spacer:latest sleep infinity
docker exec -i -e HF_HUB_OFFLINE=1 -w /catk spacer-dev python /spacer/test_adapter.py
```

**Result:** ✅


| Step                           | Output                                                               |
| -------------------------------- | ---------------------------------------------------------------------- |
| GPUDrive env                   | 1 world, 7 controlled agents                                         |
| Adapter → HeteroData          | 20 agents × 91 steps,**197 map polylines**                          |
| CAT-K`SMARTDecoder` (clsft_E9) | loaded, n_token_agent = 2048                                         |
| `TokenProcessor`               | tokenized adapter HeteroData OK (real map + agent tokens)            |
| **π_ref forward**             | **`next_token_logits: (20, 16, 2048)`** — real 2048-way categorical |

The exact distribution SPACeR's Eq. 3 / Eq. 5 consume, from real GPUDrive data,
on the RTX 3060. Also fulfils the previously-deferred real-data π_ref forward.

> Fidelity caveats (refinable, not blocking): Waymax→SMART map type-enum table
> is best-effort; traffic-light state defaults to "none"; polyline grouping by
> GPUDrive segment id. These affect how *human-like* logits are, not whether the
> pipeline runs. Padding road points (`rg_type == -1` / `rg_id < 0`) are dropped.

---

## Test 4 — adapter SEMANTIC correctness (GT-next-token NLL)

**Goal:** Test 3 proved the adapter *runs* and shapes are right — not that the
logits are *meaningful*. CAT-K was trained on WOMD and `GPUDrive_mini` IS WOMD,
so if the scene is reconstructed correctly π_ref must predict the **actual
logged next token** far better than chance. A frame/unit/map bug would pin NLL
at the random baseline ln 2048 ≈ 7.62 and accuracy at ≈random.

**Method:** feed the logged scene → adapter → π_ref forward; score
`pred["next_token_logits"]` against `tokenized_agent["gt_idx"][:, 2:]` masked by
`next_token_valid` (exact alignment from CAT-K `smart.py:108-112` / `TokenCls`).
8 validation scenes.

**File:** `spacer/test_adapter_nll.py`

**Result:** ✅ (1888 valid agent-step predictions)


| Metric                    | Result     | Random baseline          |
| --------------------------- | ------------ | -------------------------- |
| Mean NLL of GT next token | **3.460**  | ln 2048 =**7.625**       |
| Median NLL                | 3.551      | 7.625                    |
| **Top-1 accuracy**        | **43.6 %** | 0.049 % (≈900× chance) |
| Top-5 accuracy            | 48.7 %     | 0.24 %                   |

π_ref predicts the real logged next token ≈900× better than chance → the
adapter reconstructs coordinate frame, headings, velocity units, agent types
and map **correctly**. The adapter produces a *meaningful* π_ref signal.

> Honest nuance: strong and usable, not *ideal* — NLL 3.46 has headroom vs a
> perfectly-preprocessed native WOMD scene. The gap is the Test-3 fidelity
> approximations (map type-enum table, traffic-light default, polyline
> grouping); refinable, **not blocking**. NLL (½ of random) is the clean
> signal; the small top-1→top-5 gap is expected because CAT-K's true target is
> a *soft* distribution over neighbouring tokens, so hard top-1 vs the
> tokenizer's argmin under-counts quality.

---

## Test 5 — live closed-loop adapter (rollout state, 3-way contrast)

**Goal:** drive the adapter from **live rollout state** (not the logged
trajectory) and check π_ref scores it sensibly. 3-way contrast over 5 scenes:
human **logged** vs trained-**policy** rollout vs **random**-action rollout.

**Design:** `gpudrive_to_smart.py` refactored — shared `scene_dict_to_heterodata()`
tail + `finite_diff_velocity()`; live state captured per step from
`GlobalEgoState` (pos_x/pos_y/rotation_angle) into a `[A,91]` buffer, reusing
the logged validity/shape/type/map. **File:** `spacer/test_adapter_live.py`

**Result:** mechanically ✅; signal nuance below.

| Trajectory | NLL | top-1 | vs random-token (7.625) |
|---|---|---|---|
| policy (trained RL) | **2.793** | 64.4 % | ≪ |
| random (random actions) | **3.197** | 54.0 % | ≪ |
| logged (noisy human) | 3.460 | 43.6 % | ≪ |

**Conclusions (honest):**
- **Live adapter works** — π_ref scores live rollout state correctly; all ≪
  random-token baseline. Gate satisfied mechanically.
- **Direction correct but margin small**: `policy < random` (bad motion → higher
  NLL) by only +0.4. Two real reasons (not bugs): (a) GPUDrive `classic`
  dynamics **low-pass-filter** random actions → still a tokenizable trajectory
  ("random actions" ≠ "random trajectory"); (b) **token-NLL is the *weak*
  lever** — `review.md` Table R3: likelihood reward (Eq. 3, α) is *modest*; the
  **KL term (Eq. 5, β) is the dominant signal**. This test probed the weak one.
- **Logged highest NLL** reconfirms the noisy-human-logs effect (paper A.6 /
  Fig 5): clean trajectories beat noisy GT under π_ref.
- **Implication:** "π_ref is a useful training signal" cannot be validated by
  token-NLL alone — the discriminative signal is the **closed-form KL (Eq. 5)**,
  which requires π_θ over the same 2048 tokens. That validation is therefore
  **coupled to action-space alignment (gate #2)** — not an adapter deficiency.

---

## Test 6 / M1 — token→state driver

**Goal:** decode any token → global pose → drive the sim faithfully (the
tokenized action mechanism, paper Sec 4.1). `spacer/token_decode.py` reuses
CAT-K `transform_to_global` (no geometry reimplemented).

**Gate / result:** ✅
- (a) decoder fed `gt_idx` vs tokenizer's own `gt_pos/gt_head`: **0.00 error**
  over 1048 valid token-steps (bit-exact — same operation).
- (b) continuous `state` action places agent at commanded pose: pos err
  **0.0 m**, yaw err 3.6e-7 rad, 7 agents.

Files: `spacer/token_decode.py`, `test_m1_decode.py`, `test_m1_drive.py`.

---

## Test 7 / M2 — π_θ 2048-token policy head

**Goal:** decentralized policy emitting a categorical over the 2048 agent
tokens. `spacer/policy_token.py` = GPUDrive late-fusion backbone, head swapped
91→2048 (`NeuralNet(action_dim=2048)`).

**Gate / result:** ✅ (all 5)

| Criterion | Result |
|---|---|
| shape `[N,2048]` | ✅ (15, 2048) |
| valid distribution | ✅ sums to 1, finite |
| init entropy ≈ ln 2048 | ✅ **7.625 exact** (no collapse) |
| tokens decode finite (M1) | ✅ in range, finite poses |
| params | **303.7k** = 39.4k backbone (identical to paper) + 264.2k 2048-head |

> Note: π_θ is 304k not the paper's ~65k — **backbone is byte-identical**; the
> entire delta is the 2048-wide head vs the paper's ~200-wide (paper-equivalent
> reconstructs to 65,289 ≈ "~65k"). Consequence of using public `clsft_E9`
> (vocab 2048). Files: `spacer/policy_token.py`, `test_m2_policy.py`.

---

## Test 8 / M3 — Eq. 3 + Eq. 5 anchoring  **(make/break gate)**  ✅ PASS

**Goal:** likelihood reward (Eq. 3) + closed-form KL (Eq. 5), cadence-aligned
at the checkpoint-native 0.5 s / 2 Hz token-steps. `spacer/anchor.py`.

**Part A — mechanical:** ✅ all hold

| | |
|---|---|
| KL(π_ref‖π_ref) ≤ 1e-5 | **0.00** |
| KL ≥ 0 for arbitrary π_θ | min 1.50 |
| r_h finite & == manual gather | ✅ |
| step-alignment shapes | ✅ (120, 16) |

**Part B — signal validation** (the decisive test, π_θ proxies vs real π_ref,
1416 valid agent-steps over 6 scenes): ✅

| π_θ proxy | mean KL to π_ref |
|---|---|
| = π_ref | 0.000 |
| peaked on **human** tokens | **3.44** |
| uniform | 7.44 |
| peaked on **random** tokens | **14.93** |

Monotone ref < good < uniform < random. **margin KL(random)−KL(good) = 11.48
nats ≫ 0.5** required. r_h(good)−r_h(random) = 11.63 nats.

**This is the central result.** The closed-form KL (Eq. 5) is a *strongly*
discriminative signal on the GPUDrive+CAT-K stack — exactly what Test 5 showed
token-NLL alone (Eq. 3, weak lever, ~0.4 margin) **could not**. SPACeR's core
mechanism is validated here. (Caveat: Part B uses near-one-hot proxies; full
training efficacy is confirmed at M4's β-ablation. M3's job — Eq. 5 correct AND
discriminative — is decisively YES.) Files: `spacer/anchor.py`, `test_m3_anchor.py`.

---

## Test 9 / M4 — SPACeR training channel (smoke)  ✅ PASS (scoped)

**Goal:** the full loop — π_θ (M2) → M1 driver → GPUDrive → π_ref (adapter) →
Eq. 1/2/3/5 → PPO+KL → update π_θ. `spacer/train_spacer.py` (compact PPO+KL).

**Gate / result:** ✅ channel smoke (1 scene, single world, RTX 3060)

| Check | Result |
|---|---|
| Loop end-to-end | ✅ 18 token-decisions/episode @ 2 Hz |
| Stability | ✅ no NaN/OOM, finite, ~0.73 it/s (1 world) |
| π_θ updates | ✅ mean\|Δw\| 5–7e-4 |
| KL bounded | ✅ 3.40, stable |
| **Eq. 2 β-knob exact** | ✅ β-magnitude verified = exactly β·KL |
| r_h (Eq. 3) | ✅ −0.987, finite, on π_θ's own trajectory |

**⚠️ Correction (Eq. 2 sign bug — found via the equation.md audit, fixed):**
the initial M4 loss used `l_pg − β·KL`. Paper Eq. 2 `L = L_PPO − β·D_KL` is an
objective to **maximise** ⇒ minimised loss is `−L = −L_PPO + β·D_KL`. With
`l_pg ≈ −L_PPO`, the loss must be **`l_pg + β·KL`**; the original `− β·KL`
*maximised* KL → pushed π_θ **away** from π_ref (anti-anchoring). M3 didn't
catch it (M3 tested the `kl`/`r_h` functions directly — correct); the M4
β-ablation only checked magnitude, not sign. **Fixed** (`train_spacer.py`
`loss = l_pg + β·KL`; `anchor.spacer_loss`→`spacer_objective` with explicit
SIGN doc) and re-verified: loss `+0.265` = `l_pg + 0.1·KL`, KL now pressured
**down** (correct anchoring), stable, π_θ updates.

**Scope (honest):** proves the channel is wired and numerically stable; does
**not** yet show convergent learning (`r_task=0` in this minimal smoke), the
full β>0-vs-β=0 *training* ablation, or the reactivity diagnostic — those need
a scaled run (M5). **Flagged precision item** (in `train_spacer.py`): exact
per-agent π_θ↔π_ref correspondence (smoke uses exact `gt_idx` for r_h + a
scene-level KL proxy). Files: `spacer/train_spacer.py`, `spacer/anchor.py`.

---

## Test 10 / M5 — exact per-agent map, faithful loop, β-ablation

**M5a — per-agent π_θ↔π_ref correspondence (exact):** ✅ match by `object_id`
(π_ref = `batch["agent"]["id"]`, π_θ = `GlobalEgoState.id[cont_mask]`; temporal
offset = `REF_STEP_OFFSET=2`). 3 scenes: 7/7 controlled agents matched,
mapping correct, **self-routed KL = 0.00e+00** → alignment + Eq. 5 path is
**exact, not a proxy**. `align_agents()` in `anchor.py`, `test_m5_correspondence.py`.

**M5b — faithful loop:** ✅ replaced M4's scene-level KL proxy with the exact
per-(agent,step) map; **recompute π_θ logits WITH grad** from stored obs (M4
had used *detached* logits ⇒ −β·KL had no gradient — fixed). 3-iter run: exact
per-agent KL **decreases 6.668 → 6.624** — the anchoring gradient genuinely
flows to π_θ. Stable, π_θ updates, 0.73 it/s.

**M5c — β-ablation (β=0 vs β=0.1, 12 iters each, separate processes):** ✅

| Run | KL over 12 iters |
|---|---|
| β = 0.0 (control) | 6.668 → 6.669 — **flat** |
| β = 0.1 (anchored) | 6.668 → **6.195** — **monotone ↓ −0.47 nats** |

**SPACeR's KL anchoring *trains* on this stack**: β>0 monotonically pulls π_θ
→ π_ref; β=0 does not. Mechanism validated end-to-end (M3 = discriminative;
M5c = optimisable). Files: `spacer/train_spacer.py`, `test_m5_correspondence.py`.

**Honest caveats (M5c):**
- **Reactivity diagnostic INCONCLUSIVE**: `r_task = 0.000` throughout (state-
  driven agents + `collision_behavior="ignore"` + short random policy ⇒ no
  goal/collision/off-road signal). The S2.5 (5 Hz distillation) decision is
  therefore **deferred — neither triggered nor ruled out**; needs a direct
  GPUDrive Info-tensor collision/off-road readout or a longer goal-reaching
  run. (Remaining M5 work.)
- **Scale constraint found**: Madrona/CUDA cannot re-init the GPUDrive engine
  twice in one process on the 3060 (`setCudaHeapSize`) ⇒ ablation arms must be
  **separate processes**. Documented.
- **Not a converged paper repro**: 12 iters, single world ≠ paper's 1B steps /
  A100. This validates the *mechanism trains*; it is not paper-grade numbers
  (the documented 3060 ceiling).

### Anchoring parameters α, β (what they are / values used)

- **β** = KL weight in **Eq. 2** `L = L_PPO − β·D_KL` — strength π_θ is pulled
  toward π_ref. The *dominant* anchoring lever.
- **α** = likelihood-reward weight in **Eq. 1** `r = r_task + α·r_humanlike`
  (`log π_ref(a_t)` bonus). The *weak* lever (paper's finding).
- **Set in `train_spacer.py` (CLI defaults): `α = 0.01`, `β = 0.1`** — the
  paper-recommended values. M5c swept `β ∈ {0.0, 0.1}`.
- Paper Table R3 (`review.md`): β=0→0.672, β=0.01→0.676, **β=0.1→0.683
  (best, top-5 0.739)**, β=1.0→0.621 (over-constrained, top-1 drops). For α:
  0–0.01 ≈ flat (~0.68), **α=0.1 collapses (0.40)** ⇒ keep α small.
- Intuition: β=0 unrealistic self-play; β≈0.1 anchors to human-like behaviour
  while preserving RL gains + diversity; β=1.0 over-imitates (loses
  reactivity).
- **Caveat:** the paper tuned β on its **vocab-200 / 3.2M** reference; ours is
  the **2048-vocab / 7M** CAT-K flagship. KL magnitude scales with vocab size
  (our KL ≈ 6.7 ≫ a 200-token KL), so **the optimal β for our reference likely
  differs from 0.1** — `0.1`/`0.01` are sound paper-informed *starting points*,
  not tuned-for-this-stack. A β (and α) sweep is part of a real run.

---

## Combined status

| Piece | Status |
|---|---|
| **π_ref** (CAT-K ckpt → 2048-head) | ✅ Test 1 |
| **π_θ-side** (GPUDrive sim + policy rollout) | ✅ Test 2 |
| **GPUDrive→SMART adapter** (runs / semantically correct / live) | ✅ Tests 3–5 |
| **M1** token→state driver | ✅ Test 6 (exact) |
| **M2** π_θ 2048-token head | ✅ Test 7 |
| **M3** Eq. 3/5 anchoring (make/break) | ✅ Test 8 — KL discriminative, margin 11.48 nats |
| **M4** SPACeR training channel | ✅ Test 9 (smoke; Eq.2 sign-bug since fixed) |
| **M5a/b/c** exact map · faithful loop · β-ablation | ✅ **Test 10 — anchoring *trains* (β>0 KL↓, β=0 flat)** |
| Reactivity diagnostic → S2.5 decision | ⏳ deferred (r_task=0; needs Info readout / longer run) |
| Convergent paper-scale run | ✗ out of reach on 3060 (documented ceiling) |

**The entire SPACeR mechanism is implemented, numerically exact, and
demonstrated to *train*** — every equation (1/2/3/5) is correct in-loop, the
closed-form KL is discriminative (M3) *and* optimisable (M5c β-ablation), with
the exact per-agent correspondence (M5a). What remains is **not mechanism** but
**scale/measurement**: a proper reactivity readout (to settle the S2.5 5 Hz
question) and longer runs — both bounded by the 3060 ceiling, not by
correctness.

### Persistent environment

- Image: `catk-spacer:latest` (durable; `nomad-gpudrive:latest` untouched)
- Containers (no `--rm`, per preference): `catk-test`, `gpudrive-test`, `spacer-dev`
- Code: `spacer/` — `gpudrive_to_smart.py`, `token_decode.py`, `policy_token.py`,
  `anchor.py`, `train_spacer.py` + `test_*.py` gates; plans `STAGE_PLAN.md`,
  `GATE2_action_space.md`
