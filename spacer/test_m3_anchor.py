"""M3 gate.

Part A (mechanical): KL≥0; KL(π_ref‖π_ref)≤1e-5; r_h finite & == manual
gather; step alignment shapes match.
Part B (signal): π_θ proxies vs the real π_ref → monotone
KL(ref)≈0 < KL(good) < KL(random) < KL(uniform); margin KL(random)−KL(good);
r_h(good) − r_h(random). PASS = Part A all hold AND monotone AND
KL(random)−KL(good) ≥ 0.5 nats.
"""
import sys, math, dataclasses, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from gpudrive_to_smart import gpudrive_to_heterodata
from anchor import kl_theta_ref, r_humanlike, align_executed_tokens

dev = "cuda"
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=1, dataset_size=6, sample_with_replacement=False)
ec = dataclasses.replace(
    EnvConfig(), ego_state=cfg.ego_state, road_map_obs=cfg.road_map_obs,
    partner_obs=cfg.partner_obs, reward_type=cfg.reward_type, norm_obs=cfg.norm_obs,
    dynamics_model=cfg.dynamics_model, collision_behavior=cfg.collision_behavior,
    dist_to_goal_threshold=cfg.dist_to_goal_threshold,
    polyline_reduction_threshold=cfg.polyline_reduction_threshold,
    remove_non_vehicles=cfg.remove_non_vehicles, lidar_obs=cfg.lidar_obs,
    disable_classic_obs=cfg.lidar_obs, obs_radius=cfg.obs_radius,
    steer_actions=torch.round(torch.linspace(-torch.pi, torch.pi, cfg.action_space_steer_disc), decimals=3),
    accel_actions=torch.round(torch.linspace(-4.0, 4.0, cfg.action_space_accel_disc), decimals=3),
)
env = GPUDriveTorchEnv(config=ec, data_loader=loader,
                       max_cont_agents=cfg.max_controlled_agents, device=dev)
ck = torch.load("/ckpt/clsft_E9.ckpt", map_location="cpu", weights_only=False)
mc = OmegaConf.create(ck["hyper_parameters"]).model_config
tp = TokenProcessor(**mc.token_processor)
dec = SMARTDecoder(**mc.decoder, n_token_agent=tp.n_token_agent).to(dev).eval()
dec.load_state_dict({k[8:]: v for k, v in ck["state_dict"].items()
                     if k.startswith("encoder.")}, strict=True)
NT = tp.n_token_agent

# accumulate real π_ref logits + executed (GT) tokens over scenes
L, G, V = [], [], []
for _ in range(6):
    env.reset()
    b = Batch.from_data_list([gpudrive_to_heterodata(env, 0)])
    tmap, tag = tp(b)
    to = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    with torch.no_grad():
        pred = dec(to(tmap), to(tag))
    L.append(pred["next_token_logits"])                    # [A,16,2048] = π_ref
    G.append(align_executed_tokens(tag["gt_idx"]).to(dev)) # [A,16]
    V.append(pred["next_token_valid"].bool())              # [A,16]
ref = torch.cat(L); gt = torch.cat(G); val = torch.cat(V)
print(f"π_ref logits {tuple(ref.shape)} | valid (agent,step) {int(val.sum())}")

# ---------- Part A : mechanical ----------
kl_self, kl_self_m = kl_theta_ref(ref, ref, val)                 # π_θ ← π_ref
rnd_logits = torch.randn_like(ref)
kl_rnd, _ = kl_theta_ref(rnd_logits, ref, val)
rh, rh_m = r_humanlike(ref, gt, val)
manual = torch.log_softmax(ref.float(), -1).gather(-1, gt.long().unsqueeze(-1)).squeeze(-1)
A1 = float(kl_self[val].max()) <= 1e-5
A2 = bool((kl_rnd[val] >= -1e-6).all())
A3 = torch.isfinite(rh[val]).all().item() and torch.allclose(rh[val], manual[val], atol=1e-5)
A4 = ref.shape[:2] == gt.shape == val.shape
print(f"[A1] KL(π_ref‖π_ref)≤1e-5      : {A1}  (max {float(kl_self[val].max()):.2e})")
print(f"[A2] KL≥0 for arbitrary π_θ    : {A2}  (min {float(kl_rnd[val].min()):.2e})")
print(f"[A3] r_h finite & ==gather     : {A3}")
print(f"[A4] step alignment shapes     : {A4}  ({tuple(ref.shape[:2])})")
partA = A1 and A2 and A3 and A4

# ---------- Part B : signal validation (π_θ proxies) ----------
def peaked(idx, hi=12.0):                    # near-one-hot logits at idx
    z = torch.zeros_like(ref); z.scatter_(-1, idx.long().unsqueeze(-1), hi); return z
rand_tok = torch.randint(0, NT, gt.shape, device=dev)
_, kl_ref = kl_theta_ref(ref, ref, val)                       # ≈0
_, kl_good = kl_theta_ref(peaked(gt), ref, val)               # π_θ peaks on human tokens
_, kl_rand = kl_theta_ref(peaked(rand_tok), ref, val)         # π_θ peaks on random tokens
_, kl_unif = kl_theta_ref(torch.zeros_like(ref), ref, val)    # π_θ uniform
_, rh_good = r_humanlike(ref, gt, val)
_, rh_rand = r_humanlike(ref, rand_tok, val)
kr, kg, ku, k0 = float(kl_rand), float(kl_good), float(kl_unif), float(kl_ref)
# correct expected ordering: identity≈0 < good (human tokens) is the lowest
# non-trivial; a π_θ peaked on a π_ref-implausible token is the WORST (higher
# than uniform). So: ref < good < uniform < random.
mono = (k0 < kg) and (kg < ku) and (ku < kr)
margin = kr - kg
rh_gap = float(rh_good) - float(rh_rand)
print(f"\nKL(ref‖ref)={k0:.3f}  KL(good)={kg:.3f}  "
      f"KL(uniform)={ku:.3f}  KL(random)={kr:.3f}")
print(f"monotone ref<good<uniform<random : {mono}")
print(f"margin KL(random)−KL(good)       : {margin:.3f} nats  (need ≥0.5)")
print(f"r_h(good)−r_h(random)            : {rh_gap:.3f} nats")
partB = mono and margin >= 0.5

print("\nVERDICT:",
      "M3 PASS — Eq.3/5 correct (Part A) and KL is a discriminative signal "
      f"(Part B, margin {margin:.2f}≥0.5)" if (partA and partB) else
      f"M3: Part A={'ok' if partA else 'FAIL'}, Part B margin={margin:.2f} "
      f"({'ok' if partB else '<0.5 — documented finding, see STAGE_PLAN on-fail'})")
