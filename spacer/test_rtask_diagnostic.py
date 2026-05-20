"""Diagnostic for the r_task = 0 mystery (Variant 4 config).

Tests three hypotheses for why `env.get_rewards()` was 0 in M4/M5c:
  H1 — `state`-dynamics teleport BYPASSES physics event detection
       (events never fire under teleport ⇒ r_task=0 by construction)
  H2 — `collision_behavior="ignore"` zeroes the *penalty* itself
       (events fire but don't enter the weighted reward)
  H3 — random/policy motion just doesn't trigger events in a short rollout
       (events would fire under sufficiently bad actions)

Method
------
1. Introspect `gpudrive.datatypes.info.Info` attributes (we don't know the
   exact event-flag field names a-priori).
2. Run a short rollout and print, per sub-step:
     - env.get_rewards() per controlled agent (the value we read as r_task)
     - all boolean-like Info fields (the raw event flags)
3. Three CLI modes that, taken together, distinguish H1/H2/H3:

     # baseline: state + ignore + gentle motion (reproduces M5c r_task=0)
     python /spacer/test_rtask_diagnostic.py
     # H3 control: state + ignore but DELIBERATELY bad motion (large jumps)
     python /spacer/test_rtask_diagnostic.py --bad
     # H2 control: state + STOP (penalty should now fire if event detection works)
     python /spacer/test_rtask_diagnostic.py --bad --collision_behavior stop

Reading the matrix
------------------
  baseline   --bad      --bad+stop      ⇒ Diagnosis
  events:0   events:0   events:0        ⇒ H1 (state teleport bypasses physics)
  events:0   events>0   events>0        ⇒ H3 (just need worse motion)
  events:0   events>0   reward<0        ⇒ H2 (ignore zeroes penalty; stop fires it)
  events>0   …          …               ⇒ events fire fine; M5c r_task=0 was
                                          just random-policy / short-rollout luck
"""
import sys, dataclasses, argparse, torch
sys.path.insert(0, "/catk"); sys.path.insert(0, "/spacer")
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.info import Info

DEV = "cuda"

ap = argparse.ArgumentParser()
ap.add_argument("--collision_behavior", choices=["ignore", "stop", "remove"],
                default="ignore")
ap.add_argument("--bad", action="store_true",
                help="drive agents in large jumps to maximise event probability")
ap.add_argument("--steps", type=int, default=12, help="sim steps to drive")
args = ap.parse_args()

cfg = load_config("/gpd/examples/experimental/config/reliable_agents_params")
loader = SceneDataLoader(root="/gpd/data/processed/validation",
                         batch_size=1, dataset_size=1, sample_with_replacement=False)
ec = dataclasses.replace(
    EnvConfig(),
    dynamics_model="state",
    collision_behavior=args.collision_behavior,
    remove_non_vehicles=cfg.remove_non_vehicles,
    obs_radius=cfg.obs_radius,
    reward_type="weighted_combination",
    goal_achieved_weight=0.0,            # Variant 4
    collision_weight=-0.75,
    off_road_weight=-0.75,
)
env = GPUDriveTorchEnv(config=ec, data_loader=loader,
                       max_cont_agents=cfg.max_controlled_agents, device=DEV)
env.reset()
cmask = env.cont_agent_mask
print(f"[config] dynamics=state collision_behavior={args.collision_behavior} "
      f"bad-motion={args.bad} steps={args.steps} | controlled={int(cmask.sum())}")

# --- 1. Introspect Info schema -----------------------------------------------
info0 = Info.from_tensor(env.sim.info_tensor(), backend="torch", device=DEV)
attrs = [a for a in sorted(dir(info0))
         if not a.startswith("_") and not callable(getattr(info0, a))]
print("[Info attrs]")
for a in attrs:
    v = getattr(info0, a)
    if torch.is_tensor(v):
        print(f"   {a:24s} shape {str(tuple(v.shape)):<14}  dtype {v.dtype}")
flag_attrs = []
for a in attrs:
    v = getattr(info0, a)
    if not torch.is_tensor(v) or v.ndim < 2:
        continue
    # heuristic: bool or small-int field is likely an event flag
    if v.dtype == torch.bool or (v.dtype in (torch.uint8, torch.int8,
                                             torch.int32, torch.int64)
                                 and int(v.max()) <= 1):
        flag_attrs.append(a)
print(f"[flag-like Info attrs] {flag_attrs}\n")

# --- 2. Rollout --------------------------------------------------------------
g0 = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                backend="torch", device=DEV)
ix = g0.pos_x[0].clone(); iy = g0.pos_y[0].clone()
iz = g0.pos_z[0].clone(); ih = g0.rotation_angle[0].clone()

any_reward_nonzero = False
any_flag_fired = {a: False for a in flag_attrs}

for t in range(args.steps):
    W, A = cmask.shape
    g = GlobalEgoState.from_tensor(env.sim.absolute_self_observation_tensor(),
                                   backend="torch", device=DEV)
    act = torch.zeros((W, A, 10), dtype=torch.float32, device=DEV)
    if args.bad:
        # drive +5 m / step in x, +2 m / step in y, rotate — agents will fly
        # off the road and possibly through each other
        act[0, :, 0] = ix + (t + 1) * 5.0
        act[0, :, 1] = iy + (t + 1) * 2.0
        act[0, :, 2] = iz
        act[0, :, 3] = ih + 0.3 * (t + 1)
    else:
        # gentle: nudge +0.5 m / step in x, keep heading
        act[0, :, 0] = g.pos_x[0] + 0.5
        act[0, :, 1] = g.pos_y[0]
        act[0, :, 2] = g.pos_z[0]
        act[0, :, 3] = g.rotation_angle[0]
    env.step_dynamics(act)
    r = env.get_rewards()[cmask]
    if float(r.abs().sum()) > 0:
        any_reward_nonzero = True
    info = Info.from_tensor(env.sim.info_tensor(), backend="torch", device=DEV)
    parts = [f"t{t:02d}", f"r_task μ={r.mean():+.3f} "
             f"(min={r.min():+.2f} max={r.max():+.2f})"]
    for a in flag_attrs:
        v = getattr(info, a)
        try:
            vv = v[0][cmask[0]]
        except Exception:
            continue
        n_fired = int(vv.bool().sum())
        if n_fired:
            any_flag_fired[a] = True
            parts.append(f"{a}={n_fired}/{int(cmask.sum())}")
    print(" | ".join(parts))

# --- 3. Summary --------------------------------------------------------------
print()
print(f"[summary] any r_task ≠ 0 over rollout : {any_reward_nonzero}")
for a, fired in any_flag_fired.items():
    print(f"          flag '{a}' ever fired   : {fired}")
print(f"          mode: dynamics=state collision_behavior={args.collision_behavior} "
      f"bad-motion={args.bad}")
