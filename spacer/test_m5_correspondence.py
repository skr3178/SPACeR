"""M5 (precision item): exact per-agent π_θ↔π_ref correspondence.

Replaces the M4-smoke scene-level KL proxy with an exact per-(agent,step) map.
Validates:
 1. id intersection non-empty between π_θ (env controlled) and π_ref (adapter)
 2. mapping correctness: theta_ids[gather][valid] == ref_ids[valid]
 3. exact KL path shape/finiteness/≥0 over the matched, next_token_valid mask
 4. self-consistency: route ref_logits through theta-space and back via the map
    ⇒ KL ≈ 0  (proves the alignment + Eq.5 path is EXACT, not a proxy)
"""
import sys, dataclasses, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.datatypes.observation import GlobalEgoState
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from gpudrive_to_smart import gpudrive_to_heterodata
from anchor import align_agents, kl_theta_ref, align_executed_tokens

dev = "cuda"
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=1, dataset_size=3, sample_with_replacement=False)
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

ok = True
for sc in range(3):
    env.reset()
    cmask = env.cont_agent_mask
    ego = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                     backend="torch", device=dev)
    theta_ids = ego.id[0][cmask[0]].long()                  # π_θ logits row order
    b = Batch.from_data_list([gpudrive_to_heterodata(env, 0)])
    tmap, tag = tp(b)
    to = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    with torch.no_grad():
        pred = dec(to(tmap), to(tag))
    ref_logits = pred["next_token_logits"]                  # [A_ref,16,2048]
    vtok = pred["next_token_valid"].bool()                  # [A_ref,16]
    ref_ids = b["agent"]["id"].to(dev).long()               # [A_ref]
    A_ref, T, V = ref_logits.shape
    A_th = int(theta_ids.numel())

    gather, valid = align_agents(ref_ids, theta_ids)
    inter = int(valid.sum())
    map_ok = bool((theta_ids[gather][valid] == ref_ids[valid]).all()) if inter else False

    # exact KL path: a θ tensor in θ-space → gather to ref order
    theta_space = torch.randn(A_th, T, V, device=dev)
    theta_aligned = theta_space[gather]                      # [A_ref,16,2048]
    m = vtok & valid.unsqueeze(1)
    kl, klm = kl_theta_ref(theta_aligned, ref_logits, m)
    shape_ok = tuple(kl.shape) == (A_ref, T) and torch.isfinite(klm).item() \
        and bool((kl[m] >= -1e-6).all())

    # self-consistency: scatter ref_logits into θ-space, gather back ⇒ KL≈0
    back = torch.zeros(A_th, T, V, device=dev)
    back[gather[valid]] = ref_logits[valid]
    kl_self, klm_self = kl_theta_ref(back[gather], ref_logits, m)
    exact = float(kl_self[m].max()) <= 1e-5 if m.any() else True

    print(f"scene {sc}: A_ref={A_ref} A_θ={A_th} matched={inter} | "
          f"map_ok={map_ok} shape_ok={shape_ok} self-KL_max="
          f"{float(kl_self[m].max()) if m.any() else 0.0:.2e} exact={exact}")
    ok &= (inter > 0) and map_ok and shape_ok and exact

print("VERDICT:", "M5 CORRESPONDENCE OK — exact per-agent π_θ↔π_ref map "
      "(id-matched, KL path exact, not a proxy)" if ok else
      "FAIL — correspondence/exactness broken")
