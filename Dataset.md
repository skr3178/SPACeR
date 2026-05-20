
# Dataset Files

https://huggingface.co/datasets/EMERGE-lab/GPUDrive_mini   (small, ~1k scenes)


https://huggingface.co/datasets/EMERGE-lab/GPUDrive        (full)

---

# Datasets Used in the Paper (SPACeR, arXiv:2510.18060)

## 1. Waymo Open Motion Dataset (WOMD)
The single underlying dataset for all experiments (Ettinger et al., 2021). Each
scenario spans 9 seconds; SPACeR initializes at 1s and simulates the remaining 8s.
- Project / download portal: https://waymo.com/open/
- Download (motion, scenario protobuf format) — requires accepting the Waymo
  license, then via `gsutil`:
  `gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/`
- Paper website / docs: https://waymo.com/open/data/motion/
- Reference: Ettinger et al., "Large Scale Interactive Motion Forecasting for
  Autonomous Driving: The Waymo Open Motion Dataset", ICCV 2021.

### Usage in the paper
- **SPACeR / PPO / HR-PPO training:** 10k resampled WOMD scenarios, run through
  the GPUDrive simulator (Kazemkhani et al., 2024), 600 parallel worlds, up to
  64 controlled agents per rollout.
- **SMART / CAT-K baselines:** trained on the full WOMD training dataset
  (~500k scenarios).
- **Evaluation:** WOSAC (Waymo Open Sim Agents Challenge) validation protocol,
  computed on a 2% validation subset.

## 2. GPUDrive processed WOMD scenes (recommended for reproduction)
GPUDrive distributes a processed/JSON-format derivative of WOMD under the same
Waymo license. These are the links already listed above:
- https://huggingface.co/datasets/EMERGE-lab/GPUDrive_mini   (small, ~1k scenes)
- https://huggingface.co/datasets/EMERGE-lab/GPUDrive        (full)
- GPUDrive code: https://github.com/Emerge-Lab/gpudrive
- Reference: Kazemkhani et al., "GPUDrive: Data-driven, multi-agent driving
  simulation at 1 million FPS", ICLR 2025 (arXiv:2408.01584).

## Related benchmark (not a separate dataset)
- **WOSAC** — Waymo Open Sim Agents Challenge (Montali et al., NeurIPS 2023,
  arXiv:2305.12032). An evaluation protocol built on WOMD, not a separate
  dataset download. https://waymo.com/open/challenges/2024/sim-agents/

> Note: WOMD requires a signed Waymo license agreement and cannot be
> redistributed; the GPUDrive HuggingFace datasets are the practical way to
> obtain the exact processed scenes SPACeR uses.

---

# Downloaded Models (local)

Two kinds of model have been downloaded locally. They play **different roles**
in SPACeR — do not confuse them.

## A. CAT-K / SMART reference checkpoints  → these are SPACeR's π_ref

- **Location:** `SPACeR/checkpoints/`
- **Source:** CAT-K, https://github.com/NVlabs/catk
  (Zhang et al., "Closed-Loop Supervised Fine-Tuning of Tokenized Traffic
  Models", CVPR 2025, arXiv:2412.05334)

| File | Size | What it is | Paper role |
|------|------|------------|------------|
| `pre_bc_E31.ckpt` | 85 MB | SMART base, **behavior-cloning** pretrained (epoch 31, 195k steps) | the paper's **"SMART"** reference (BC-only, pre fine-tuning) |
| `clsft_E9.ckpt`   | 71 MB | **CAT-K** closed-loop supervised fine-tuned (epoch 9, 61k steps) | the paper's **"CAT-K"** reference — primary **π_ref** in main experiments |

- Both: ~7M params (811 tensors), single `encoder` module root,
  PyTorch-Lightning **2.4.0** + Hydra / OmegaConf checkpoints
  (`pip install omegaconf` required just to unpickle).

### Checkpoint analysis (from the OmegaConf `hyper_parameters`)

Both checkpoints share the **identical SMART decoder architecture**; they
differ only in the training stage (BC pretrain vs. CAT-K closed-loop SFT).

**Shared architecture (`model_config.decoder`):**

| Field | Value |
|-------|-------|
| `hidden_dim` | 128 |
| `num_heads` / `head_dim` | 8 / 16 |
| `num_freq_bands` | 64 |
| `num_map_layers` / `num_agent_layers` | 3 / 6 |
| `pl2pl_radius` / `pl2a_radius` / `a2a_radius` | 10 / 30 / 60 m |
| `time_span` | 30 |
| `num_historical_steps` / `num_future_steps` | 11 / 80 |
| `dropout` / `hist_drop_prob` | 0.1 / 0.1 |

**Shared tokenizer (`model_config.token_processor`) — this defines 𝒜:**

- `agent_token_file: agent_vocab_555_s2.pkl` ← the **agent trajectory-token
  vocabulary**, **2048 tokens** per agent class (veh/ped/cyc), each a
  `(2048, 6, 4, 2)` motion template (6 future steps × 4 bbox corners × xy).
  The `555` in the filename is a build label, **not** the vocab size; verified
  via `token_processor.py:49` `n_token_agent = agent_token_all_veh.shape[0] = 2048`
  and the checkpoint's `token_predict_head` (`out_features = 2048`). This is the
  full, irreducible action space 𝒜 π_θ must align to for Eqs. 3 & 5 — **all
  2048 are valid centroids; do not filter/truncate** (that would corrupt the
  trained reference distribution and require retraining π_ref).
- `map_token_file: map_traj_token5.pkl` (map tokenization)
- sampling at inference: `validation_rollout_sampling = topk_prob, num_k=5`

**Stage-specific differences:**

| | `pre_bc_E31.ckpt` (SMART, BC) | `clsft_E9.ckpt` (CAT-K, SFT) |
|---|---|---|
| `epoch` / `global_step` | 31 / 194,829 | 9 / 60,886 |
| `model_config.finetune` | `false` | `true` |
| `lr` | 5.0e-4 | 5.0e-5 |
| `lr_total_steps` / `lr_min_ratio` | 64 / 0.01 | 32 / 0.05 |
| `training_rollout_sampling.criterium` | `topk_prob` (`num_k=-1`, full) | `topk_prob_sampled_with_dist` (`num_k=32`, `temp=1e-5`, **k-annealing** `k_min=32`, `end_epoch=10`) |
| `training_loss.label_smoothing` | 0.1 | 0.0 |
| extra `training_loss` (SFT only) | — | `target_weighting=td_1.0`, `scheduler_epoch=5`, `look_ahead_steps=0`, `linear_interp_dagger=false`, `bptt_ignore_dynamics=false` |

The `clsft_E9` config is the **signature of CAT-K closed-loop fine-tuning**:
`topk_prob_sampled_with_dist` rollout sampling with K-annealing (Closest-Among-
Top-K) on top of the BC-pretrained SMART weights — exactly the method of
Zhang et al. (2025) and the stronger π_ref the SPACeR paper uses by default.
- **Role:** these are the *centralized tokenized reference model* π_ref. They
  supply the human-likeness signals SPACeR anchors to:
  - log-likelihood reward — Eq. (3): `r_humanlike = log π_ref(a_t | s_t)`
  - closed-form KL — Eq. (5): `D_KL(π_θ ‖ π_ref)` over the shared token vocab.
- **Recommended:** use `clsft_E9.ckpt` (CAT-K) as the primary π_ref — the
  stronger prior the paper uses by default. Keep `pre_bc_E31.ckpt` for the
  "SMART vs CAT-K reference" ablation.
- **How to use:** load through CAT-K's own `SMART` LightningModule from the
  cloned repo (`reference_code/catk/src/`), then run **forward-only** (no
  gradients, no autoregressive sampling) to obtain the per-timestep categorical
  over the **2048-token** agent vocabulary. Sketch:

  ```python
  # from within reference_code/catk
  import torch
  from src.smart.model import SMART          # CAT-K LightningModule
  ckpt = "../../checkpoints/clsft_E9.ckpt"
  model = SMART.load_from_checkpoint(ckpt, map_location="cuda").eval()
  with torch.no_grad():
      logits = model(scene_tokens)           # → categorical over 2048 agent tokens
      logp   = torch.log_softmax(logits, -1) # feeds Eq.3 reward / Eq.5 KL
  ```

  (CAT-K resolves data/checkpoint paths via its Hydra config in
  `reference_code/catk/configs/`; set `WOMD_VAL_DIR` etc. before running its
  own eval scripts. For SPACeR you only need the forward pass.)

## B. GPUDrive self-play policy  → a *baseline*, NOT a reference model

- **Location:** `reference_code/gpudrive/models/policy_S10_000_02_27/`
  (`model.safetensors` ~207 KB, `config.json`, `README.md`)
- **Source:** https://huggingface.co/daphne-cornelisse/policy_S10_000_02_27
  (Cornelisse et al., "Building reliable sim driving agents by scaling
  self-play", arXiv:2502.14706) — "Best Policy", trained on 10k WOMD scenarios.
- **Role:** a ~65k-param **decentralized self-play policy** (the PPO / HR-PPO
  family of *baseline*). It is **not** π_ref — it does not output a categorical
  over the shared trajectory-token vocabulary, so it cannot provide the Eq.3 /
  Eq.5 anchoring signals. Use it as a reference *baseline to compare against*,
  or as a warm-start for π_θ.
- **How to use:**

  ```python
  from gpudrive.networks.late_fusion import NeuralNet
  # via Hugging Face hub:
  agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")
  # or from the local copy:
  agent = NeuralNet.from_pretrained("models/policy_S10_000_02_27")
  ```

  Trained with `examples/experimental/config/reliable_agents_params.yaml` —
  changing env/observation configs degrades its performance. See GPUDrive
  tutorial `examples/tutorials/04_use_pretrained_sim_agent.ipynb`.

> Summary: **π_ref = `checkpoints/clsft_E9.ckpt` (CAT-K).** The GPUDrive
> `policy_S10_000_02_27` is a self-play baseline, not the reference.

---

# Additional references

3WOMD download instructions available at https://waymo.com/intl/en_us/open/download.
4https://protobuf.dev/


The WOMD [22] dataset itself is licensed under a non-commercial license (www.
waymo.com/open/terms) and the evaluation code for our Waymo Open Sim
Agents Challenge (WOSAC) is released under a BSD+limited patent license. See
20
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/PATENTS and


https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/LICENSE

Code available at https://github.com/wangwenxi-handsome/Joint-Multipathpp.