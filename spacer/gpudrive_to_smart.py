"""GPUDrive -> SMART/CAT-K HeteroData adapter.

Builds the exact intermediate dicts that CAT-K's own
`src.data_preprocess.get_agent_features` and `src.smart.utils.preprocess.preprocess_map`
consume, from live GPUDrive simulator state, then assembles the `HeteroData` that
`TokenProcessor.forward(data)` expects (mirrors WaymoTargetBuilderVal: HeteroData(dict)).

Design principle: reuse CAT-K preprocessing functions; this module only does the
GPUDrive-state -> WOMD-style-dict extraction. Nothing in catk/ is modified.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import torch
from torch_geometric.data import HeteroData

# `src.data_preprocess` imports waymo_open_dataset at module top (used only by
# its WOMD-parser fns, NOT by get_agent_features which is pure numpy/scipy).
# Stub it so we can reuse get_agent_features without the TF/waymo conflict stack.
if "waymo_open_dataset" not in sys.modules:
    _wod = types.ModuleType("waymo_open_dataset")
    _protos = types.ModuleType("waymo_open_dataset.protos")
    _protos.scenario_pb2 = types.SimpleNamespace()
    _wod.protos = _protos
    sys.modules["waymo_open_dataset"] = _wod
    sys.modules["waymo_open_dataset.protos"] = _protos

# CAT-K reusable preprocessing (catk repo must be on sys.path, cwd=/catk)
from src.data_preprocess import get_agent_features
from src.smart.utils.preprocess import preprocess_map

# GPUDrive datatypes
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints, MapElementIds
from gpudrive.datatypes.observation import GlobalEgoState

NUM_STEPS = 91            # WOMD / GPUDrive TRAJ_LEN
NUM_HIST_STEPS = 11       # SMART num_historical_steps (step 10 = "current")
STEP_CURRENT = 10

# GPUDrive EntityType -> SMART agent type {0:veh, 1:ped, 2:cyc}
_ENTITY_TO_SMART = {1: 0, 2: 1, 3: 2}

# Waymax MapElementIds -> (SMART point_type 0-9, SMART polygon_type 0-3)
# polygon types: 0 lane, 1 road_edge, 2 road_line, 3 crosswalk
_MAP_ELEM_TO_SMART = {
    MapElementIds.LANE_FREEWAY: (0, 0),
    MapElementIds.LANE_SURFACE_STREET: (1, 0),
    MapElementIds.LANE_BIKE_LANE: (3, 0),
    MapElementIds.STOP_SIGN: (2, 0),
    MapElementIds.ROAD_EDGE_UNKNOWN: (4, 1),
    MapElementIds.ROAD_EDGE_BOUNDARY: (4, 1),
    MapElementIds.ROAD_EDGE_MEDIAN: (5, 1),
    MapElementIds.ROAD_LINE_UNKNOWN: (6, 2),
    MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE: (6, 2),
    MapElementIds.ROAD_LINE_BROKEN_SINGLE_YELLOW: (6, 2),
    MapElementIds.ROAD_LINE_BROKEN_DOUBLE_YELLOW: (6, 2),
    MapElementIds.ROAD_LINE_SOLID_SINGLE_WHITE: (7, 2),
    MapElementIds.ROAD_LINE_SOLID_SINGLE_YELLOW: (7, 2),
    MapElementIds.ROAD_LINE_SOLID_DOUBLE_WHITE: (8, 2),
    MapElementIds.ROAD_LINE_SOLID_DOUBLE_YELLOW: (8, 2),
    MapElementIds.ROAD_LINE_PASSING_DOUBLE_YELLOW: (8, 2),
    MapElementIds.CROSSWALK: (9, 3),
    MapElementIds.SPEED_BUMP: (9, 3),
    MapElementIds.DRIVEWAY: (9, 3),
}


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def extract_gpudrive_scene(env, world_idx: int = 0, device: str = "cuda") -> dict:
    """Pull agent trajectories + road graph for one world, in the **sim-native
    (mean-centered) frame**.

    Frame note: GPUDrive's simulator — `set_state` input, `GlobalEgoState`
    output, collision/off-road detection — all operate in the mean-centered
    frame. We deliberately do NOT `restore_mean()` here so that decoded poses
    fed back via `set_state` land in GPUDrive's own coordinates. (An earlier
    version restored the mean → global frame; that teleported every agent by
    ~world_mean on the first token-step, silently disabling off-road
    detection. See test.md.) `mean_xy` is still returned so any caller that
    genuinely wants the global frame can add it back.
    """
    means_xy = env.sim.world_means_tensor().to_torch()[:, :2]
    mx, my = float(means_xy[world_idx, 0]), float(means_xy[world_idx, 1])

    # --- agents: full 91-step logged trajectory, sim-native (mean-centered) ---
    log = LogTrajectory.from_tensor(
        env.sim.expert_trajectory_tensor(),
        env.num_worlds, env.max_agent_count, backend="torch",
    )
    pos_xy = _to_np(log.pos_xy[world_idx])              # [A, 91, 2]
    vel_xy = _to_np(log.vel_xy[world_idx])              # [A, 91, 2]
    yaw = _to_np(log.yaw[world_idx]).reshape(-1, NUM_STEPS)        # [A, 91]
    valid = _to_np(log.valids[world_idx]).reshape(-1, NUM_STEPS).astype(bool)  # [A,91]

    ego = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(), backend="torch", device=device,
    )
    length = _to_np(ego.vehicle_length[world_idx]).reshape(-1)
    width = _to_np(ego.vehicle_width[world_idx]).reshape(-1)
    height = _to_np(ego.vehicle_height[world_idx]).reshape(-1)
    posz = _to_np(ego.pos_z[world_idx]).reshape(-1)
    obj_id = _to_np(ego.id[world_idx]).reshape(-1).astype(np.int64)

    info = env.sim.info_tensor().to_torch()[world_idx]   # [A, 5]
    entity = _to_np(info[:, 4]).astype(int)
    cmask = _to_np(env.cont_agent_mask[world_idx]).astype(bool)

    # --- road graph: sim-native (mean-centered) points + waymax type + id ---
    # No restore_mean — same frame as the agents above, so the adapter scene
    # is internally consistent and matches GPUDrive's own road graph.
    rg = GlobalRoadGraphPoints.from_tensor(
        env.sim.map_observation_tensor(), backend="torch", device=device,
    )
    rg_x = _to_np(rg.x[world_idx]).reshape(-1)
    rg_y = _to_np(rg.y[world_idx]).reshape(-1)
    rg_type = _to_np(rg.vbd_type[world_idx]).reshape(-1).astype(int)
    rg_id = _to_np(rg.id[world_idx]).reshape(-1).astype(np.int64)

    return dict(
        pos_xy=pos_xy, vel_xy=vel_xy, yaw=yaw, valid=valid,
        length=length, width=width, height=height, posz=posz,
        obj_id=obj_id, entity=entity, cmask=cmask,
        rg_x=rg_x, rg_y=rg_y, rg_type=rg_type, rg_id=rg_id,
        mean_xy=(mx, my),
    )


def build_track_infos(s: dict) -> dict:
    """WOMD-style track_infos for CAT-K get_agent_features.
    states layout [A,91,9] = x,y,z, length,width,height, heading, vx,vy
    """
    A = s["pos_xy"].shape[0]
    states = np.zeros((A, NUM_STEPS, 9), dtype=np.float32)
    states[:, :, 0:2] = s["pos_xy"]
    states[:, :, 2] = s["posz"][:, None]
    states[:, :, 3] = s["length"][:, None]
    states[:, :, 4] = s["width"][:, None]
    states[:, :, 5] = s["height"][:, None]
    states[:, :, 6] = s["yaw"]
    states[:, :, 7:9] = s["vel_xy"]

    obj_type = np.array([_ENTITY_TO_SMART.get(int(e), 0) for e in s["entity"]],
                        dtype=np.uint8)
    role = np.zeros((A, 3), dtype=bool)
    ego_idx = np.where(s["cmask"] & s["valid"][:, STEP_CURRENT])[0]
    if len(ego_idx) == 0:
        ego_idx = np.where(s["valid"][:, STEP_CURRENT])[0]
    if len(ego_idx) > 0:
        role[ego_idx[0], 0] = True          # one designated ego (av_index)
    role[s["cmask"], 2] = True              # controlled -> "predict"

    return dict(
        object_id=s["obj_id"], object_type=obj_type, role=role,
        valid=s["valid"], states=states,
    )


def build_map_data(s: dict) -> dict:
    """Group GPUDrive road points by segment id into polylines and emit the
    map_data dict consumed by CAT-K preprocess_map()."""
    # drop padding road points (UNKNOWN == -1, or sentinel ids)
    keep = (s["rg_type"] >= 0) & (s["rg_id"] >= 0)
    rg_id, rg_type = s["rg_id"][keep], s["rg_type"][keep]
    rg_x, rg_y = s["rg_x"][keep], s["rg_y"][keep]

    seg_ids = np.unique(rg_id)
    pt_pos, pt_type, poly_type, poly_light = [], [], [], []
    n_poly = 0
    for sid in seg_ids:
        m = rg_id == sid
        if m.sum() < 2:
            continue
        wtype = int(np.bincount(rg_type[m]).argmax())  # dominant waymax type
        mapping = _MAP_ELEM_TO_SMART.get(
            MapElementIds(wtype) if wtype in MapElementIds._value2member_map_
            else MapElementIds.LANE_SURFACE_STREET,
            (1, 0),
        )
        ptype, pltype = mapping
        xy = np.stack([rg_x[m], rg_y[m]], axis=1).astype(np.float32)
        k = xy.shape[0]
        pt_pos.append(xy)
        pt_type.append(np.full(k, ptype, dtype=np.uint8))
        poly_type.append(pltype)
        poly_light.append(0)                 # traffic-light state not extracted
        n_poly += 1

    if n_poly == 0:
        # preprocess_map handles the empty case, but give it a valid structure
        return {
            "map_polygon": {"num_nodes": 0,
                            "type": torch.zeros(0, dtype=torch.uint8),
                            "light_type": torch.zeros(0, dtype=torch.uint8)},
            "map_point": {"num_nodes": 0,
                          "position": torch.zeros(0, 2, dtype=torch.float32),
                          "type": torch.zeros(0, dtype=torch.uint8)},
            ("map_point", "to", "map_polygon"): {
                "edge_index": torch.zeros(2, 0, dtype=torch.long)},
        }

    positions = np.concatenate(pt_pos, axis=0)
    types = np.concatenate(pt_type, axis=0)
    counts = [p.shape[0] for p in pt_pos]
    src = torch.arange(positions.shape[0], dtype=torch.long)
    dst = torch.arange(n_poly, dtype=torch.long).repeat_interleave(
        torch.tensor(counts))
    return {
        "map_polygon": {
            "num_nodes": n_poly,
            "type": torch.tensor(poly_type, dtype=torch.uint8),
            "light_type": torch.tensor(poly_light, dtype=torch.uint8),
        },
        "map_point": {
            "num_nodes": int(positions.shape[0]),
            "position": torch.from_numpy(positions),       # [P, 2]
            "type": torch.from_numpy(types),               # [P]
        },
        ("map_point", "to", "map_polygon"): {
            "edge_index": torch.stack([src, dst], dim=0),
        },
    }


def finite_diff_velocity(pos_xy: np.ndarray, valid: np.ndarray,
                         dt: float = 0.1) -> np.ndarray:
    """Central-difference velocity [A,T,2] from positions; dt = 0.1s @ 10 Hz.
    get_agent_features re-interpolates over valid steps, so edge handling is
    only a seed."""
    v = np.zeros_like(pos_xy, dtype=np.float32)
    v[:, 1:-1] = (pos_xy[:, 2:] - pos_xy[:, :-2]) / (2 * dt)
    v[:, 0] = (pos_xy[:, 1] - pos_xy[:, 0]) / dt
    v[:, -1] = (pos_xy[:, -1] - pos_xy[:, -2]) / dt
    v[~valid] = 0.0
    return v


def scene_dict_to_heterodata(s: dict, scenario_id: str = "gpudrive",
                             split: str = "val") -> HeteroData:
    """Shared tail: scene dict -> HeteroData (logged OR live rollout source).
    `s` must carry pos_xy/yaw/vel_xy/valid/length/width/height/posz/obj_id/
    entity/cmask + rg_*."""
    track_infos = build_track_infos(s)
    agent = get_agent_features(
        track_infos, split=split,
        num_historical_steps=NUM_HIST_STEPS, num_steps=NUM_STEPS,
    )
    map_out = preprocess_map(build_map_data(s))
    return HeteroData({
        "scenario_id": scenario_id,
        "agent": agent,
        "map_save": map_out["map_save"],
        "pt_token": map_out["pt_token"],
    })


def gpudrive_to_heterodata(env, world_idx: int = 0, split: str = "val") -> HeteroData:
    """Full adapter from the LOGGED trajectory: GPUDrive world -> HeteroData."""
    s = extract_gpudrive_scene(env, world_idx)
    return scene_dict_to_heterodata(s, f"gpudrive_world_{world_idx}", split)
