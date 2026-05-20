# SPACeR — Paper Equations

Extracted from `SPACeR.pdf` (Self-Play Anchoring with Centralized Reference Models, ICLR 2026).

---

## Section 3.1 — Problem Formulation

**Action selection and environment transition** (inline, p.4)

$$a_t^i \in \mathcal{A}, \qquad a_t^i \sim \pi_\theta(a_t^i \mid o_t^i), \qquad s_{t+1} \sim T(s_t, a_t^1, \dots, a_t^N)$$

**Eq. (1) — Total reward**

$$r_t \;=\; r_t^{\text{task}} \;+\; \alpha\, r^{\text{humanlike}}(s_t, a_t)$$

**Eq. (2) — Training objective**

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{PPO}}\big(\theta;\, A[r]\big) \;-\; \beta\, D_{\text{KL}}\big(\pi_\theta(\cdot \mid o_t)\,\big\|\,\pi_{\text{ref}}(\cdot \mid s_t)\big)$$

Eq. (2) combines three components:
- **Task performance:** $\mathcal{L}_{\text{PPO}}(\theta)$ is the standard PPO objective.
- **Human-likeness reward:** Eq. (3).
- **Distributional alignment:** the KL divergence term $D_{\text{KL}}\big(\pi_\theta(\cdot \mid o_t)\,\|\,\pi_{\text{ref}}(\cdot \mid s_t)\big)$.

**Eq. (3) — Human-likeness reward**

$$r_{\text{humanlike}}(s_t, a_t) = \log \pi_{\text{ref}}(a_t \mid s_t)$$

---

## Section 3.2 — Pretrained Reference Tokenized Model

**Eq. (4) — Conditional independence factorization across agents**

$$p(a_t \mid a_{<t},\, c) = \prod_{i=1}^{N} p\big(a_t^i \mid a_{<t},\, c\big)$$

where $a_t = (a_t^1, \dots, a_t^N)$, $t$ is the discrete timestep, and $N$ is the number of agents in the scene.

**Eq. (5) — KL divergence in closed form (aligned action space)**

$$D_{\text{KL}}\big(\pi_\theta(\cdot \mid o_t)\,\big\|\,\pi_{\text{ref}}(\cdot \mid s_t)\big) = \sum_{a \in \mathcal{A}} \pi_\theta(a \mid o_t) \, \log \frac{\pi_\theta(a \mid o_t)}{\pi_{\text{ref}}(a \mid s_t)}$$

---

## Section 4.1 — Reward Formulation

**Task reward** (unnumbered, p.6)

$$r^{\text{task}}(s_t, a_t) \;=\; w_{\text{goal}}\,\mathbb{1}[\text{Goal achieved}] \;-\; w_{\text{collided}}\,\mathbb{1}[\text{Collided}] \;-\; w_{\text{offroad}}\,\mathbb{1}[\text{Offroad}] \;+\; w_{\text{humanlike}}\, r^{\text{humanlike}}(s_t, a_t)$$

where $\mathbb{1}[\cdot] \in \{0, 1\}$. By default, $w_{\text{collided}} = w_{\text{offroad}} = 0.75$.

---

## Appendix A.5 — Waymo Sim Agent Challenge Metrics

**Eq. (6) — Negative log-likelihood of ground-truth outcome**

$$\text{NLL}(i, a, t, j) = -\log\big( p_{i,j,a}\big(F_j(x^*(i, a, t))\big) \big)$$

where $p_{i,j,a}(\cdot)$ is the empirical distribution constructed from the simulated samples, and $x^*(i, a, t)$ is the true trajectory at time $t$ (target agent $a$, scenario $i$, statistic $F_j$). Lower values indicate the simulation better reflects observed behavior.

**Eq. (7) — Per-agent summary (aggregate over valid timesteps)**

$$m(a, i, j) = \exp\!\left( -\,\frac{1}{N(i, a)} \sum_t v(i, a, t)\, \text{NLL}(i, a, t, j) \right)$$

where $N(i, a) = \sum_t v(i, a, t)$ is the number of valid timesteps for agent $a$.

**Eq. (8) — Scenario-level score (average across evaluated agents)**

$$m(i, j) = \frac{1}{A_{\text{target}}} \sum_a m(a, i, j)$$

with $A_{\text{target}}$ denoting the number of target agents in the scenario.
