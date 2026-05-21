"""World-count ceiling probe for the RTX 3060 (12 GB).

One process = one W. Madrona/CUDA cannot re-init in-process, so a sweep must
spawn a fresh process per W (see world_sweep.sh driver).

Each probe: build the env at W worlds, run 2 full `spacer_iteration` steps
(rollout + π_ref score + KL + backward + opt.step — the real training-memory
peak), then report torch's peak allocation. OOM → exit 7.

torch.cuda.max_memory_allocated tracks ONLY torch tensors — GPUDrive's Madrona
sim memory is invisible to it. The driver additionally samples nvidia-smi
memory.used (whole-GPU truth) in parallel.

  python world_sweep.py --worlds 64
"""
import os
# Match train_spacer.py: must be set before torch import. This is the
# documented fix for the W=64 fragmentation OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse

import torch

sys.path.insert(0, "/catk")
sys.path.insert(0, "/spacer")
from train_spacer import (build_env, load_ref, spacer_iteration,
                          TokenPolicy, DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, required=True)
    ap.add_argument("--iters", type=int, default=2,
                    help="spacer_iteration steps (peak hit by iter 2).")
    a = ap.parse_args()
    W = a.worlds

    try:
        torch.cuda.reset_peak_memory_stats()
        env, _ = build_env(W, n_worlds=W)
        obs0 = env.reset()
        odim = obs0[env.cont_agent_mask].shape[-1]
        policy = TokenPolicy(obs_dim=odim).to(DEV)
        tp, dec = load_ref()
        policy._ttraj = tp._get_agent_shape_and_token_traj(
            torch.zeros(env.cont_agent_mask.shape[1], dtype=torch.long))[2]
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        for _ in range(a.iters):
            spacer_iteration(env, policy, opt, tp, dec, 0.0, 0.1)
        torch.cuda.synchronize()
        alloc = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
        print(f"WSWEEP_OK W={W} torch_peak_alloc_GB={alloc:.2f} "
              f"torch_peak_reserved_GB={resv:.2f}")
    except torch.cuda.OutOfMemoryError:
        print(f"WSWEEP_OOM W={W}")
        sys.exit(7)
    except Exception as e:
        print(f"WSWEEP_ERR W={W} {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
