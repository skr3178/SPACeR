"""M2: pi_theta — tokenized self-play policy.

SPACeR's pi_theta is the GPUDrive late-fusion MLP with its actor head swapped
from 91 (13x7 accel/steer) to the **2048 agent-token vocabulary**, so its
action space matches pi_ref (cadence/vocab invariant — see STAGE_PLAN.md).
The GPUDrive `NeuralNet` is already parameterised by `action_dim`, so the
backbone is reused verbatim; only `action_dim=2048`.

Beyond the baseline's `forward` (action/logprob/entropy/value, PPO-ready),
M3's closed-form KL `D_KL(pi_theta||pi_ref) = Σ_{2048} π_θ·log(π_θ/π_ref)`
needs the **full categorical**, so we expose `logits()` / `log_probs()` /
`distribution()`.

Vehicles-first: single 2048 vocab. Per-agent-type (veh/ped/cyc each 2048) is
a later extension (route obs by type to per-type heads).

----------------------------------------------------------------------------
ARCHITECTURE  (late-fusion MLP — paper's backbone, vocab-2048 actor head)
----------------------------------------------------------------------------
input obs  [N, 2984]   (ego_state ‖ partner_obs(×63) ‖ road_map(×?))
   │
   ├──> ego_embed       :  Linear(ego_dim → 64) → tanh → Linear(64 → 64)
   ├──> partner_embed   :  Linear(partner_dim → 64) → tanh → Linear(64 → 64)
   │                       → max-pool across partner agents       [N, 64]
   └──> road_map_embed  :  Linear(roadgraph_dim → 64) → tanh → Linear(64 → 64)
                           → max-pool across road points          [N, 64]
                                                  │ concat
                                                  ▼
                                          [N, 192]  (= 3 × 64)
                                                  │
                                  shared_embed:  Linear(192 → 128) → tanh
                                                                       │
                                                                       ▼
                                                              hidden [N, 128]
                                                                ┌──────┴──────┐
                                                                ▼             ▼
                                  actor: Linear(128 → 2048)    critic: Linear(128 → 1)
                                  ← 2048-way categorical       ← state value
                                    (SPACeR head; baseline was 91)

----------------------------------------------------------------------------
PARAMETER BREAKDOWN  (verified empirically by `policy.num_params()`)
----------------------------------------------------------------------------
  Backbone (ego + partner + road_map + shared encoders)    :    39,360
  Actor head   Linear(128, 2048)  (128·2048 + 2048)        :   264,192
  Critic head  Linear(128,    1)  (128·1    +    1)        :       129
  ────────────────────────────────────────────────────────────────────
  Total π_θ                                                :   303,681   (≈304 k)

  Paper-equivalent (same backbone + 200-vocab actor)       :    65,289   (≈65 k)
  ⇒ The 5× delta is ENTIRELY the wider actor head (2048 vs ≈200);
    the backbone is byte-identical to the paper. The wider head is the
    locked consequence of using public `clsft_E9` (vocab 2048).
    See STAGE_PLAN.md S2.6 for the optional token-cluster path that
    shrinks the head back toward paper-size without retraining π_ref.

Hyperparams (defaults; built from `NeuralNet(action_dim=2048)`):
  input_dim=64, hidden_dim=128, dropout=0.0, act="tanh",
  max_controlled_agents=64, obs_dim=2984.
Init: PufferLib std=0.01 on the actor ⇒ near-uniform at init
  (entropy ≈ ln 2048 = 7.625; verified in M2 / test_m2_policy.py).
"""
import torch
import torch.nn as nn
from gpudrive.networks.late_fusion import NeuralNet

N_TOKENS = 2048  # agent token vocabulary (clsft_E9; matches pi_ref)


class TokenPolicy(nn.Module):
    def __init__(self, obs_dim: int = 2984, hidden_dim: int = 128,
                 n_tokens: int = N_TOKENS, reward_type: str = "weighted_combination",
                 vbd_in_obs: bool = False):
        super().__init__()
        self.n_tokens = n_tokens
        # exact GPUDrive backbone; only the actor head differs (action_dim).
        # NeuralNet needs a config to set vbd_in_obs / reward offsets.
        self.net = NeuralNet(
            action_dim=n_tokens, hidden_dim=hidden_dim, obs_dim=obs_dim,
            dropout=0.01,                       # paper A.3 (NeuralNet default 0.0)
            config={"reward_type": reward_type, "vbd_in_obs": vbd_in_obs},
        )

    # --- PPO-compatible path (same signature as the baseline policy) ---
    def forward(self, obs, action=None, deterministic=False):
        """Returns (token_idx, logprob, entropy, value)."""
        return self.net(obs, action=action, deterministic=deterministic)

    # --- single-forward PPO+KL update path (used by _ppo_update with K-acc) -
    # Avoids the duplicate forward through backbone+actor that calling
    # `policy(o, action=a)` then `policy.logits(o)` would do. Roughly halves
    # the PPO-update activation memory at high accum_k. Math is identical.
    def forward_with_logits(self, obs, action):
        """Single backbone forward → (newlp, entropy, value, log_probs)
        all derived from one shared-embed pass. `log_probs` is the
        log_softmax over the 2048-token vocab — the caller uses it for
        closed-form KL (Eq. 5) without a second forward."""
        hidden = self.net.encode_observations(obs)
        logits = self.net.actor(hidden)
        value  = self.net.critic(hidden).reshape(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        newlp = log_probs.gather(-1,
                                 action.long().unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return newlp, entropy, value, log_probs

    # --- full categorical (needed by M3 Eq. 5 KL / Eq. 3) ---
    def logits(self, obs):
        return self.net.actor(self.net.encode_observations(obs))  # [N, 2048]

    def log_probs(self, obs):
        return torch.log_softmax(self.logits(obs), dim=-1)         # [N, 2048]

    def distribution(self, obs):
        return torch.distributions.Categorical(logits=self.logits(obs))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
