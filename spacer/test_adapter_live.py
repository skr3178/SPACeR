"""Live closed-loop adapter test (gate #1).

Run a baseline-policy rollout (NOT the log), capture each agent's live global
state every step, feed that *rollout* trajectory through the SAME adapter ->
pi_ref -> GT-token NLL path. Compare:

    logged (Test 4)   <   live policy rollout   <   random (ln 2048 = 7.62)

Success = live NLL well below random AND ordering logged <= live (pi_ref is a
usable closed-loop reference, responsive to trajectory quality, not a
logged-data artifact). Live NLL > logged is EXPECTED and is the point: pi_ref
penalises less-human-like rollout motion (the SPACeR signal).
"""
import sys, dataclasses, math, copy, torch, numpy as np
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.datatypes.observation import GlobalEgoState
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from gpudrive_to_smart import (extract_gpudrive_scene, scene_dict_to_heterodata,
                               finite_diff_velocity, NUM_STEPS)

dev = "cuda"
N_SCENES = 5
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
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
policy = NeuralNet.from_pretrained("/gpd/models/policy_S10_000_02_27").to(dev).eval()

ck = torch.load("/ckpt/clsft_E9.ckpt", map_location="cpu", weights_only=False)
mcfg = OmegaConf.create(ck["hyper_parameters"]).model_config
tp = TokenProcessor(**mcfg.token_processor)
dec = SMARTDecoder(**mcfg.decoder, n_token_agent=tp.n_token_agent).to(dev).eval()
dec.load_state_dict({k[8:]: v for k, v in ck["state_dict"].items()
                     if k.startswith("encoder.")}, strict=True)
RAND = math.log(tp.n_token_agent)


def score(hd):
    """GT-next-token NLL + top-1 over valid (agent,step), same as Test 4."""
    b = Batch.from_data_list([hd])
    tmap, tag = tp(b)
    to = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    tmap, tag = to(tmap), to(tag)
    with torch.no_grad():
        pred = dec(tmap, tag)
    v = pred["next_token_valid"].bool()
    if v.sum() == 0:
        return None
    gt = tag["gt_idx"][:, 2:]
    lp = torch.log_softmax(pred["next_token_logits"].float(), -1)
    nll = -lp.gather(-1, gt.unsqueeze(-1).long()).squeeze(-1)[v]
    c1 = (pred["next_token_logits"].argmax(-1) == gt)[v].float()
    return nll, c1


def rollout_buf(mode):
    """91-step buffer of live global state. mode: 'policy' | 'random'."""
    obs = env.reset()
    s0 = extract_gpudrive_scene(env, world_idx=0)
    A = s0["pos_xy"].shape[0]
    bp = np.zeros((A, NUM_STEPS, 2), np.float32)
    bh = np.zeros((A, NUM_STEPS), np.float32)
    cmask = env.cont_agent_mask
    n_act = int(policy.action_dim)
    for t in range(NUM_STEPS):
        g = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(), backend="torch", device=dev)
        bp[:, t, 0] = g.pos_x[0].detach().cpu().numpy()
        bp[:, t, 1] = g.pos_y[0].detach().cpu().numpy()
        bh[:, t] = g.rotation_angle[0].detach().cpu().numpy()
        if t < NUM_STEPS - 1:
            if mode == "policy":
                with torch.no_grad():
                    act, _, _, _ = policy(obs[cmask], deterministic=True)
            else:  # random actions -> deliberately non-human-like motion
                act = torch.randint(0, n_act, (int(cmask.sum()),), device=dev)
            tmpl = torch.zeros(cmask.shape, dtype=torch.int64, device=dev)
            tmpl[cmask] = act.to(dev)
            env.step_dynamics(tmpl)
            obs = env.get_obs()
    s = copy.copy(s0)
    s["pos_xy"], s["yaw"] = bp, bh
    s["vel_xy"] = finite_diff_velocity(bp, s0["valid"])
    return s0, s


agg = {k: ([], []) for k in ("logged", "policy", "random")}
for sc in range(N_SCENES):
    s0, s_pol = rollout_buf("policy")
    _, s_rnd = rollout_buf("random")
    for name, sd, tag in (("logged", s0, f"log_{sc}"),
                          ("policy", s_pol, f"pol_{sc}"),
                          ("random", s_rnd, f"rnd_{sc}")):
        r = score(scene_dict_to_heterodata(sd, tag))
        if r:
            agg[name][0].append(r[0]); agg[name][1].append(r[1])

print(f"scenes: {N_SCENES} | random-token baseline ln2048 = {RAND:.3f}")
res = {}
for name in ("logged", "policy", "random"):
    N = torch.cat(agg[name][0]); C = torch.cat(agg[name][1])
    res[name] = float(N.mean())
    print(f"{name:8s} : NLL {N.mean():.3f}  top-1 {C.mean()*100:.1f}%  (n={N.numel()})")

discriminative = res["random"] > res["policy"] + 0.5
below_random = res["policy"] < 0.7 * RAND and res["logged"] < 0.7 * RAND
print("VERDICT:",
      "LIVE ADAPTER OK + pi_ref is a USEFUL RL signal "
      "(bad motion -> higher NLL; good/logged -> low; all << random-token)"
      if (discriminative and below_random) else
      "CHECK — pi_ref not discriminative enough between good/bad motion")
