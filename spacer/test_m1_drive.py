"""M1 part b: prove the `state`-dynamics driver.

Decode (proven exact in test_m1_decode) gives global poses; this confirms that
feeding the continuous 10-vec state action
  (x, y, z, yaw, vx, vy, vz, wx, wy, wz)   -> sim.action_tensor()[:, :, :10]
actually places the agent at the commanded global pose. Command each controlled
agent a known target (current pos + 5 m in x, yaw += 0.2), step, read back
GlobalEgoState, compare realized vs commanded.
"""
import sys, dataclasses, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.datatypes.observation import GlobalEgoState

dev = "cuda"
cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=1, dataset_size=1, sample_with_replacement=False)
ec = dataclasses.replace(
    EnvConfig(),
    dynamics_model="state",                       # <-- continuous state dynamics
    collision_behavior="ignore",
    remove_non_vehicles=cfg.remove_non_vehicles,
    obs_radius=cfg.obs_radius,
)
env = GPUDriveTorchEnv(config=ec, data_loader=loader,
                       max_cont_agents=cfg.max_controlled_agents, device=dev)
env.reset()
cmask = env.cont_agent_mask                       # [W, A]
W, A = cmask.shape


def global_state():
    g = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(), backend="torch", device=dev)
    return (g.pos_x[0].clone(), g.pos_y[0].clone(),
            g.pos_z[0].clone(), g.rotation_angle[0].clone())

x0, y0, z0, h0 = global_state()

# commanded target: +5 m in x, +0.2 rad yaw, zero velocity
tx, ty, th = x0 + 5.0, y0.clone(), h0 + 0.2
act = torch.zeros((W, A, 10), dtype=torch.float32, device=dev)
act[0, :, 0] = tx
act[0, :, 1] = ty
act[0, :, 2] = z0
act[0, :, 3] = th
env.step_dynamics(act)

x1, y1, z1, h1 = global_state()
m = cmask[0].bool()
ex = (x1 - tx).abs()[m]
ey = (y1 - ty).abs()[m]
dh = torch.atan2(torch.sin(h1 - th), torch.cos(h1 - th)).abs()[m]
moved = (x1 - x0).abs()[m]

print(f"controlled agents: {int(m.sum())}")
print(f"commanded dx = +5.000 m ; realized mean |dx| = {moved.mean():.3f} m")
print(f"pos error vs commanded : x {ex.mean():.3e}  y {ey.mean():.3e}  (m)")
print(f"yaw error vs commanded : {dh.mean():.3e} rad")
ok = (ex.mean() < 0.5) and (ey.mean() < 0.5) and (dh.mean() < 0.1) \
     and (moved.mean() > 4.0)
print("VERDICT:", "M1 STATE-DRIVER OK — agents follow commanded global pose; "
      "token->state driver complete" if ok else
      "CHECK — state action did not place agent at commanded pose")
