"""M3: SPACeR anchoring — Eq. 3 (likelihood reward) + Eq. 5 (closed-form KL).

Both π_θ and π_ref are categoricals over the SAME 2048 agent tokens at the
SAME 0.5 s / 2 Hz token-step cadence (cadence invariant). π_ref comes from the
adapter→TokenProcessor→SMARTDecoder forward as `next_token_logits` [A, 16, 2048]
(token-steps (10→15)…(85→90)); the executed-token / GT index aligns as
`tokenized_agent["gt_idx"][:, 2:]` (mirrors CAT-K smart.py:108-112 / Test 4).

Eq. 5  D_KL(π_θ‖π_ref) = Σ_{a∈2048} π_θ(a) · [logπ_θ(a) − logπ_ref(a)]
Eq. 3  r_humanlike     = log π_ref(a_t | s_t)            (a_t = executed token)
"""
import torch
import torch.nn.functional as F

# π_ref next_token_logits has 16 token-steps; GT/executed index uses [:, 2:].
REF_STEP_OFFSET = 2


def align_executed_tokens(gt_idx):
    """gt_idx [A, 18] (token-steps {5,10,…,90}) → [A, 16] aligned to
    next_token_logits' (10→15)…(85→90)."""
    return gt_idx[:, REF_STEP_OFFSET:]


def align_agents(ref_ids, theta_ids):
    """Per-agent π_θ↔π_ref correspondence by object_id (M5 precision item).

    ref_ids   : [A_ref]  ids of π_ref agents (batch["agent"]["id"], the
                get_agent_features-filtered/ordered set)
    theta_ids : [A_th]   ids of π_θ agents (GlobalEgoState.id[cont_mask],
                the order policy.logits rows are produced in)
    Returns (gather [A_ref] long, valid [A_ref] bool): for each π_ref agent,
    the row index into the π_θ tensor with the same id (gather), and whether a
    match exists (valid). Rows with valid=False must be masked out of Eq.5/3.
    """
    ref_ids = ref_ids.long().view(-1)
    theta_ids = theta_ids.long().view(-1)
    # position of each id in theta (-1 if absent)
    pos = torch.full((int(ref_ids.max().item()) + 2,), -1, dtype=torch.long,
                     device=ref_ids.device) if ref_ids.numel() else \
        torch.empty(0, dtype=torch.long)
    lut = {}
    for j, tid in enumerate(theta_ids.tolist()):
        lut.setdefault(tid, j)
    gather = torch.tensor([lut.get(int(r), 0) for r in ref_ids.tolist()],
                          dtype=torch.long, device=ref_ids.device)
    valid = torch.tensor([int(r) in lut for r in ref_ids.tolist()],
                         dtype=torch.bool, device=ref_ids.device)
    return gather, valid


def kl_theta_ref(logits_theta, logits_ref, valid=None):
    """Closed-form Eq. 5 KL, per (agent, step).
    logits_theta, logits_ref: [..., 2048]; valid: [...] bool (optional).
    Returns (kl_elementwise [...], kl_mean scalar over valid)."""
    logp = F.log_softmax(logits_theta.float(), dim=-1)
    logq = F.log_softmax(logits_ref.float(), dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(-1)                       # [...]
    kl = kl.clamp_min(0.0)                                  # guard fp noise
    if valid is None:
        return kl, kl.mean()
    v = valid.bool()
    return kl, (kl[v].mean() if v.any() else kl.new_tensor(0.0))


def r_humanlike(logits_ref, executed_token, valid=None):
    """Eq. 3: log π_ref(a_t | s_t) at the executed token.
    logits_ref [..., 2048]; executed_token [...] long; valid [...] bool."""
    logq = F.log_softmax(logits_ref.float(), dim=-1)
    r = logq.gather(-1, executed_token.long().unsqueeze(-1)).squeeze(-1)  # [...]
    if valid is None:
        return r, r.mean()
    v = valid.bool()
    return r, (r[v].mean() if v.any() else r.new_tensor(0.0))


def total_reward(r_task, r_h, alpha):
    """Eq. 1: r = r_task + α · r_humanlike."""
    return r_task + alpha * r_h


def spacer_objective(l_ppo, kl_mean, beta):
    """Eq. 2 objective (to MAXIMISE):  L = L_PPO − β · D_KL.
    π_ref frozen ⇒ KL grads flow to π_θ only.

    SIGN: this is the maximisation objective exactly as in the paper. If you
    optimise by *minimising* a loss, minimise  −L = −L_PPO + β·D_KL  (i.e. the
    anchoring term enters a minimised loss with **+β**, not −β). Getting this
    sign wrong pushes π_θ *away* from π_ref (anti-anchoring)."""
    return l_ppo - beta * kl_mean
