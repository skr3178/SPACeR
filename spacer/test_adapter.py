"""End-to-end: GPUDrive_mini scene -> adapter -> TokenProcessor -> SMARTDecoder.

Confirms the GPUDrive->SMART adapter yields a HeteroData that flows through
CAT-K's real pipeline and produces the 2048-way agent-token logits (the Eq.3/5
signal for SPACeR). Run inside catk-spacer container, cwd=/catk.
"""
import sys, dataclasses, torch
sys.path.insert(0, "/catk")
sys.path.insert(0, "/spacer")

from omegaconf import OmegaConf
from torch_geometric.data import Batch

from gpudrive.networks.late_fusion import NeuralNet  # noqa (warms gpudrive)
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from gpudrive_to_smart import gpudrive_to_heterodata

dev = "cuda"

# --- 1. GPUDrive env on one mini scene ---
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=1, dataset_size=1,
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
env.reset()
print("[1] GPUDrive env ready:", env.num_worlds, "world,",
      int(env.cont_agent_mask.sum()), "controlled agents")

# --- 2. Adapter: GPUDrive -> HeteroData ---
hd = gpudrive_to_heterodata(env, world_idx=0, split="val")
print("[2] HeteroData built | agents:", hd["agent"]["position"].shape,
      "| map polylines:", hd["map_save"]["traj_pos"].shape)

# --- 3. Load CAT-K reference model (clsft_E9) ---
ck = torch.load("/ckpt/clsft_E9.ckpt", map_location="cpu", weights_only=False)
mcfg = OmegaConf.create(ck["hyper_parameters"]).model_config
tp = TokenProcessor(**mcfg.token_processor)
dec = SMARTDecoder(**mcfg.decoder, n_token_agent=tp.n_token_agent).to(dev).eval()
dec.load_state_dict({k[8:]: v for k, v in ck["state_dict"].items()
                     if k.startswith("encoder.")}, strict=True)
print("[3] CAT-K SMARTDecoder loaded | n_token_agent:", tp.n_token_agent)

# --- 4. Collate (mimic DataLoader so 'batch'/num_graphs exist) + tokenize ---
# TokenProcessor holds CPU token tensors -> tokenize on CPU, move to GPU after.
batch = Batch.from_data_list([hd])
tokenized_map, tokenized_agent = tp(batch)
to_dev = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
tokenized_map, tokenized_agent = to_dev(tokenized_map), to_dev(tokenized_agent)
print("[4] Tokenized | map keys:", sorted(tokenized_map.keys())[:5],
      "| agent keys:", sorted(tokenized_agent.keys())[:5])

# --- 5. Forward pass -> 2048-way logits ---
with torch.no_grad():
    pred = dec(tokenized_map, tokenized_agent)
print("[5] pred_dict keys:", list(pred.keys()))
for k, v in pred.items():
    if torch.is_tensor(v) and v.dim() >= 2 and 2048 in tuple(v.shape):
        print(f"    {k}: {tuple(v.shape)}  <-- 2048-way agent-token categorical")
print("VERDICT: GPUDrive scene -> adapter -> CAT-K pi_ref forward OK (real data)")
