"""Adapter CORRECTNESS test (not just "runs").

Feed the logged GPUDrive scene through the adapter, then check how well CAT-K
pi_ref predicts the *actual logged next token*. CAT-K was trained on WOMD and
GPUDrive_mini IS WOMD -> if the scene is reconstructed correctly, NLL of the
ground-truth token must be FAR below the random baseline ln(2048) ~= 7.62 and
top-1/top-5 accuracy must be high. Garbage (frame/unit/map bug) -> ~random.

Alignment mirrors CAT-K smart.py:108-112 (TokenCls):
  logits = pred["next_token_logits"]            # [n_agent, 16, 2048]
  target = tokenized_agent["gt_idx"][:, 2:]     # [n_agent, 16]
  valid  = pred["next_token_valid"]             # [n_agent, 16]
"""
import sys, dataclasses, math, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from gpudrive.networks.late_fusion import NeuralNet  # noqa
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from gpudrive_to_smart import gpudrive_to_heterodata

dev = "cuda"
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
N_SCENES = 8
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=1, dataset_size=N_SCENES,
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

ck = torch.load("/ckpt/clsft_E9.ckpt", map_location="cpu", weights_only=False)
mcfg = OmegaConf.create(ck["hyper_parameters"]).model_config
tp = TokenProcessor(**mcfg.token_processor)
dec = SMARTDecoder(**mcfg.decoder, n_token_agent=tp.n_token_agent).to(dev).eval()
dec.load_state_dict({k[8:]: v for k, v in ck["state_dict"].items()
                     if k.startswith("encoder.")}, strict=True)
RAND = math.log(tp.n_token_agent)

all_nll, all_c1, all_c5, all_n = [], [], [], 0
env.reset()
for sc in range(N_SCENES):
    hd = gpudrive_to_heterodata(env, world_idx=0, split="val")
    batch = Batch.from_data_list([hd])
    tmap, tag = tp(batch)
    to = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    tmap, tag = to(tmap), to(tag)
    with torch.no_grad():
        pred = dec(tmap, tag)
    logits = pred["next_token_logits"]                  # [A,16,2048]
    valid = pred["next_token_valid"].bool()             # [A,16]
    gt = tag["gt_idx"][:, 2:]                            # [A,16]
    if valid.sum() == 0:
        continue
    logp = torch.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, gt.unsqueeze(-1).long()).squeeze(-1)   # [A,16]
    top5 = logits.topk(5, dim=-1).indices                          # [A,16,5]
    c1 = (logits.argmax(-1) == gt)
    c5 = (top5 == gt.unsqueeze(-1)).any(-1)
    v = valid
    all_nll.append(nll[v]); all_c1.append(c1[v].float()); all_c5.append(c5[v].float())
    all_n += int(v.sum())
    env.reset()  # next scene batch

nll = torch.cat(all_nll); c1 = torch.cat(all_c1); c5 = torch.cat(all_c5)
print(f"valid (agent,step) predictions : {all_n}  over {N_SCENES} scenes")
print(f"mean NLL of GT next token      : {nll.mean():.3f}   (random baseline ln2048 = {RAND:.3f})")
print(f"median NLL                     : {nll.median():.3f}")
print(f"top-1 accuracy                 : {c1.mean()*100:.1f}%   (random ~ {100/2048:.3f}%)")
print(f"top-5 accuracy                 : {c5.mean()*100:.1f}%")
ok = (nll.mean() < 0.5 * RAND) and (c5.mean() > 0.10)
print("VERDICT:", "ADAPTER SEMANTICALLY CORRECT (pi_ref predicts real motion)"
      if ok else "SUSPECT — NLL near random => frame/unit/map reconstruction bug")
