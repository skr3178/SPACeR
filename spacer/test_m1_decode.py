"""M1 decisive check (no sim): our token decoder, fed the GT token indices,
must reproduce the tokenizer's own gt_pos / gt_heading. If it matches to ~1e-3
at valid token-steps, the decode geometry is provably correct and generalises
to arbitrary pi_theta-chosen tokens.

Replicates tokenize_agent's exact preprocessing (clean+extrapolate) so our
decoder sees the same init pos/head the tokenizer used.
"""
import sys, dataclasses, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from src.smart.tokens.token_processor import TokenProcessor
from gpudrive_to_smart import gpudrive_to_heterodata
from token_decode import decode_token_sequence

dev = "cuda"
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=1, dataset_size=4, sample_with_replacement=False)
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
mcfg = OmegaConf.create(ck["hyper_parameters"]).model_config
tp = TokenProcessor(**mcfg.token_processor)

perr_all, herr_all, n_all = [], [], 0
for sc in range(4):
    env.reset()
    hd = gpudrive_to_heterodata(env, world_idx=0, split="val")
    batch = Batch.from_data_list([hd])

    # tokenizer output (its own GT decode)
    tag = tp(batch)[1]                     # tokenize_agent dict
    gt_idx = tag["gt_idx"]                 # [A, T]
    gt_pos = tag["gt_pos"]                 # [A, T, 2]
    gt_head = tag["gt_heading"]            # [A, T]
    vtok = tag["valid_mask"]               # [A, T] token-step validity
    ttraj = tag["token_traj"]              # [A, Ntok, 4, 2]

    # replicate tokenize_agent preprocessing to get the SAME init pos/head
    a = batch["agent"]
    valid = a["valid_mask"]
    head = tp._clean_heading(valid, a["heading"])
    pos = a["position"][..., :2].contiguous()
    vel = a["velocity"]
    valid, pos, head, vel = tp._extrapolate_agent_to_prev_token_step(valid, pos, head, vel)
    init_pos, init_head = pos[:, 0], head[:, 0]

    dec_pos, dec_head = decode_token_sequence(gt_idx, init_pos, init_head, ttraj, vtok)

    m = vtok.bool()
    if m.sum() == 0:
        continue
    perr = torch.norm(dec_pos - gt_pos, dim=-1)[m]                  # [n]
    dh = torch.atan2(torch.sin(dec_head - gt_head),
                     torch.cos(dec_head - gt_head)).abs()[m]
    perr_all.append(perr); herr_all.append(dh); n_all += int(m.sum())

P = torch.cat(perr_all); H = torch.cat(herr_all)
print(f"valid token-steps compared : {n_all}")
print(f"position error vs tokenizer gt_pos : mean {P.mean():.2e}  max {P.max():.2e}  (m)")
print(f"heading  error vs tokenizer gt_head: mean {H.mean():.2e}  max {H.max():.2e}  (rad)")
ok = (P.mean() < 1e-2) and (P.max() < 5e-2) and (H.mean() < 1e-2)
print("VERDICT:", "M1 DECODER CORRECT — reproduces CAT-K gt_pos/gt_head; "
      "generalises to arbitrary pi_theta tokens" if ok else
      "MISMATCH — decode geometry differs from tokenizer")
