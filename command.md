# SPACeR — Command Log

All commands used while setting up datasets, models, and the CAT-K π_ref
test environment. Paths assume the project root:
`/media/skr/storage/self_driving/self_play/SPACeR`

---

## 1. Datasets

### GPUDrive_mini (1000 train / 150 test / 150 val processed WOMD scenes)
```bash
cd reference_code/gpudrive
HF_HUB_DISABLE_TELEMETRY=1 huggingface-cli download \
  EMERGE-lab/GPUDrive_mini --repo-type dataset --local-dir data/processed
# verify
for d in training testing validation; do \
  echo "$d: $(find data/processed/$d -name '*.json' | wc -l) json"; done
du -sh data/processed
```
(Full dataset: replace `GPUDrive_mini` with `GPUDrive`.)

---

## 2. Pretrained models

### GPUDrive self-play policy (baseline, NOT a reference model)
```bash
cd reference_code/gpudrive
HF_HUB_DISABLE_TELEMETRY=1 huggingface-cli download \
  daphne-cornelisse/policy_S10_000_02_27 \
  --local-dir models/policy_S10_000_02_27
```

### CAT-K / SMART reference checkpoints  → π_ref
Placed at `SPACeR/checkpoints/`:
- `pre_bc_E31.ckpt`  — SMART base, BC pretrained
- `clsft_E9.ckpt`    — CAT-K closed-loop SFT (primary π_ref)

---

## 3. Reference paper

```bash
cd "reference papers"
curl -sL -o "SMART - Scalable Multi-agent Real-time Motion Generation via Next-token Prediction.pdf" \
  https://arxiv.org/pdf/2405.15677
```

---

## 4. Extract open GitHub issues (NVlabs/catk)

```bash
mkdir -p reference_code/catk/issues
gh issue list --repo NVlabs/catk --state open --limit 500 \
  --json number,title,state,createdAt,updatedAt,closedAt,author,labels,assignees,milestone,comments,body,url \
  > reference_code/catk/issues/_issues_raw.json
# (then a small python script turned the JSON into per-issue .md + README.md index)
```

---

## 5. Inspect the CAT-K checkpoints (host)

```bash
pip install -q omegaconf
python3 - <<'PY'
import torch
from omegaconf import OmegaConf
for f in ['pre_bc_E31.ckpt','clsft_E9.ckpt']:
    ck = torch.load(f, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    n  = sum(v.numel() for v in sd.values() if hasattr(v,'numel'))
    print(f, '| epoch', ck.get('epoch'), '| step', ck.get('global_step'),
          '| params', f'{n/1e6:.2f}M', '| tensors', len(sd))
    print(OmegaConf.to_yaml(OmegaConf.create(ck['hyper_parameters']))[:1500])
PY
```

---

## 6. Docker — base image checks

```bash
docker images                                  # list (nomad-gpudrive:latest is the base)
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader

# verify base image: GPU + torch + GPUDrive engine
docker run --rm --gpus all nomad-gpudrive:latest python -c "
import torch, madrona_gpudrive, gpudrive
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# inspect base image deps (numpy/protobuf/tf conflict assessment)
docker run --rm nomad-gpudrive:latest python -c "
import importlib.metadata as m
for p in ['numpy','protobuf','tensorflow','torch','torch_geometric','lightning','omegaconf']:
    try: print(p, m.version(p))
    except Exception: print(p, 'NOT installed')"
```

---

## 7. Test CAT-K checkpoints in a persistent container

### 7a. Start persistent container (no --rm; checkpoints + repo mounted read-only)
```bash
docker run -d --name catk-test --gpus all \
  -v /media/skr/storage/self_driving/self_play/SPACeR/reference_code/catk:/catk:ro \
  -v /media/skr/storage/self_driving/self_play/SPACeR/checkpoints:/ckpt:ro \
  nomad-gpudrive:latest sleep infinity

docker exec catk-test python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 7b. Install minimal π_ref deps (keeps NOMAD torch 2.6/cu124 — no TF/waymo)
```bash
docker exec catk-test pip install --no-cache-dir \
  torch_geometric omegaconf \
  torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 7c. Decoupled strict load test (bypasses TF/waymo metrics chain)
```bash
docker exec -i -w /catk catk-test python - <<'PY'
import torch, sys
from omegaconf import OmegaConf
sys.path.insert(0, "/catk")
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor

for name in ["clsft_E9.ckpt", "pre_bc_E31.ckpt"]:
    ck  = torch.load(f"/ckpt/{name}", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["hyper_parameters"]).model_config
    sd  = ck["state_dict"]
    tp  = TokenProcessor(**cfg.token_processor)
    dec = SMARTDecoder(**cfg.decoder, n_token_agent=tp.n_token_agent)
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    res = dec.load_state_dict(enc, strict=True)
    print(name, "| n_token_agent", tp.n_token_agent,
          "| missing", len(res.missing_keys),
          "| unexpected", len(res.unexpected_keys))
PY
```
Result: both checkpoints → 0 missing / 0 unexpected (proper & usable as π_ref).

---

## 8. Persist the environment as a reusable image

```bash
docker commit -m "nomad-gpudrive + CAT-K pi_ref deps" catk-test catk-spacer:latest

# verify (with GPU)
docker run --rm --gpus all catk-spacer:latest python -c "
import torch, torch_geometric, torch_scatter, torch_cluster, omegaconf
import madrona_gpudrive, gpudrive
print('CAT-K deps + GPUDrive OK:', torch.cuda.get_device_name(0))"
```

### Reuse pattern (persistent, no --rm)
```bash
docker run -d --name spacer --gpus all \
  -v /media/skr/storage/self_driving/self_play/SPACeR/reference_code/catk:/catk:ro \
  -v /media/skr/storage/self_driving/self_play/SPACeR/checkpoints:/ckpt:ro \
  catk-spacer:latest sleep infinity
docker exec -it -w /catk spacer bash
```

---

## 9. M5d — multi-world rollout

Adds `--worlds N` to `train_spacer.py`: N parallel Madrona worlds per iter.
Default **W=32** (safe on the 12 GB 3060; W=64 also fits thanks to the
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set inside the script,
but is slower per iter). Pass `--worlds 1` to reproduce the pre-M5d path.

### 9a. Default run (W=32)
```bash
docker exec -w /spacer spacer-dev python train_spacer.py \
  --mode smoke --iters 5 --scenes 64
```

### 9b. Backward-compat regression (W=1)
```bash
docker exec -w /spacer spacer-dev python train_spacer.py \
  --mode smoke --iters 3 --scenes 4 --worlds 1
```

### 9c. β-ablation at W=32 (lower-variance than M5c's W=1)
```bash
docker exec -w /spacer spacer-dev python train_spacer.py \
  --mode ablate --iters 5 --scenes 64 --beta 0.1
```

### 9d. Push to ceiling (optional — slower, needs expandable_segments)
```bash
docker exec -w /spacer spacer-dev python train_spacer.py \
  --mode smoke --iters 2 --scenes 64 --worlds 64
```

### Measured on the 3060 (3 iters each, except W=32/64 = 2)
| W  | it/s | samples/iter | samples/sec |
|----|------|--------------|-------------|
| 1  | 0.73 |  ~7          | 5.1         |
| 4  | 0.30 |  ~28         | 8.4         |
| 8  | 0.25 |  ~56         | 14          |
| 16 | 0.19 | ~112         | 21          |
| 32 | 0.14 | ~224         | **31**      |
| 64 | 0.09 | ~448         | **40**      |

it/s drops with W (per-world Python adapter is serial), but PPO-relevant
samples/sec scales ~6× from W=1 to W=32 and ~8× to W=64.
