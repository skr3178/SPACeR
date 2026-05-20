"""M1: token -> global-pose decoder.

At SPACeR training time pi_theta picks an arbitrary token; we must decode it to
a global pose to drive the sim. The tokenizer only decodes its own argmin (GT)
match, so we need our own decoder. It is *exactly* CAT-K's per-step geometry in
`token_processor._match_agent_token` (lines 236-262), driven autoregressively
by a chosen token sequence instead of the argmin. Reuses CAT-K
`transform_to_global` — no geometry reimplemented.

Correctness contract: fed `gt_idx`, this must reproduce the tokenizer's own
`gt_pos` / `gt_heading` (same operation) -> see test_m1_decode.py.
"""
import torch
from src.smart.utils.rollout import transform_to_global


@torch.no_grad()
def decode_token_sequence(token_idx, init_pos, init_head, token_traj, valid_tok):
    """Autoregressive token -> pose decode.

    Args:
        token_idx : [A, T]   chosen token per agent per token-step
        init_pos  : [A, 2]   agent pos at step 0 (post clean+extrapolate)
        init_head : [A]      agent heading at step 0
        token_traj: [A, Ntok, 4, 2]  final-contour template per token (local)
        valid_tok : [A, T]   token-step validity (carry prev where invalid)
    Returns:
        pos  : [A, T, 2]     decoded global position per token-step
        head : [A, T]        decoded global heading per token-step
    """
    A, T = token_idx.shape
    ra = torch.arange(A, device=token_idx.device)
    prev_pos = init_pos.clone()
    prev_head = init_head.clone()
    out_pos, out_head = [], []
    for t in range(T):
        # all tokens -> global from current pose, pick the chosen one
        tw = transform_to_global(
            token_traj.flatten(1, 2), None, prev_pos, prev_head
        )[0].view(A, -1, 4, 2)                       # [A, Ntok, 4, 2]
        contour = tw[ra, token_idx[:, t].long()]      # [A, 4, 2]
        npos = contour.mean(1)                        # [A, 2]
        dxy = contour[:, 0] - contour[:, 3]
        nhead = torch.arctan2(dxy[:, 1], dxy[:, 0])   # [A]
        out_pos.append(npos)
        out_head.append(nhead)
        m = valid_tok[:, t]
        prev_pos = torch.where(m.unsqueeze(-1), npos, prev_pos)
        prev_head = torch.where(m, nhead, prev_head)
    return torch.stack(out_pos, 1), torch.stack(out_head, 1)
