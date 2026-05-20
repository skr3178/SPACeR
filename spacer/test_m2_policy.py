"""M2 gate: pi_theta 2048-token head on real GPUDrive obs.

PASS (all):
 1. logits shape [n_controlled, 2048]
 2. valid distribution: softmax sums to 1 (+-1e-4), no NaN/Inf
 3. init entropy ~= ln 2048 = 7.62 (+-0.3) — ~uninformative at init, no collapse
 4. sampled tokens are in [0,2048) and feed token_decode -> finite poses
 5. param count logged (backbone + 2048-head)
"""
import sys, math, dataclasses, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from omegaconf import OmegaConf
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from src.smart.tokens.token_processor import TokenProcessor
from policy_token import TokenPolicy, N_TOKENS
from token_decode import decode_token_sequence

dev = "cuda"
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=2, dataset_size=4, sample_with_replacement=False)
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
obs = env.reset()
cmask = env.cont_agent_mask
x = obs[cmask]                                  # [n_controlled, obs_dim]
obs_dim = x.shape[-1]
print(f"obs: {tuple(obs.shape)} | controlled: {x.shape[0]} | obs_dim {obs_dim}")

policy = TokenPolicy(obs_dim=obs_dim).to(dev).eval()
with torch.no_grad():
    logits = policy.logits(x)                   # [N, 2048]
    probs = torch.softmax(logits, -1)
    ent = torch.distributions.Categorical(logits=logits).entropy()
    tok, logp, entropy, val = policy(x, deterministic=False)

N = x.shape[0]
c1 = tuple(logits.shape) == (N, N_TOKENS)
c2 = torch.allclose(probs.sum(-1), torch.ones(N, device=dev), atol=1e-4) \
     and torch.isfinite(logits).all().item()
c3 = abs(float(ent.mean()) - math.log(N_TOKENS)) < 0.3
c4_range = bool(((tok >= 0) & (tok < N_TOKENS)).all())

# criterion 4: policy-sampled tokens decode to finite poses
tp = TokenProcessor(**OmegaConf.create(
    torch.load("/ckpt/clsft_E9.ckpt", map_location="cpu",
               weights_only=False)["hyper_parameters"]).model_config.token_processor)
A = N
atype = torch.zeros(A, dtype=torch.long)                       # veh
_, _, ttraj = tp._get_agent_shape_and_token_traj(atype)        # [A, 2048, 4, 2]
seq = tok.view(A, 1).cpu()                                      # one token-step
ipos = torch.zeros(A, 2); ihead = torch.zeros(A)
vmask = torch.ones(A, 1, dtype=torch.bool)
dpos, dhead = decode_token_sequence(seq, ipos, ihead, ttraj, vmask)
c4 = c4_range and torch.isfinite(dpos).all().item() and torch.isfinite(dhead).all().item()

print(f"1 shape [N,2048]      : {c1}  ({tuple(logits.shape)})")
print(f"2 valid distribution  : {c2}  (sum~1, finite)")
print(f"3 init entropy ~7.62  : {c3}  (mean {float(ent.mean()):.3f}, target {math.log(N_TOKENS):.3f})")
print(f"4 tokens decode finite: {c4}  (range ok={c4_range})")
print(f"5 params              : {policy.num_params()/1e3:.1f}k "
      f"(backbone + {N_TOKENS}-head); value head present={val.shape==(N,1)}")
print("VERDICT:", "M2 PASS — pi_theta emits a valid 2048-token categorical, "
      "decodable by the M1 driver" if (c1 and c2 and c3 and c4) else "FAIL")
