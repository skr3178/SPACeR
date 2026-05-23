# debug.md — Test 22 code audits & responses to external reviewers

Three independent external code reviews of the post-Test-22 implementation
(`train_spacer.py`, `policy_token.py`, `anchor.py`, `gpudrive_to_smart.py`,
`token_decode.py`). Each is verified claim-by-claim against the actual
code. Captured here so the audit findings are searchable without diluting
`test.md`'s experiment log.

**Top-level pattern across the three reviews:**

| | Audit #1 | Audit #2 | Audit #3 |
|---|---|---|---|
| Tone | Diagnostic / general | Surgical / line-numbered | Strategic / normalization-focused |
| Real bugs flagged | 0 | **2** (rollout history; non-controlled token-0) | 0 — but raises **KL mean-vs-sum** as a critical unknown |
| Mis-flagged "bugs" | 1 (`gt_idx` "critical bug") | 0 | 0 |
| Empirically refuted | 1 (β=1.0 recommendation) | 0 | 0 |
| Useful uninstrumented diagnostics raised | 0 | 0 | 2 (valid coverage; token distribution) |

**Combined picture of what's still genuinely worth investigating** (most
important first):

1. **KL aggregation: mean vs sum across agents** (Audit #3 Interaction B)
   — if the paper sums and we mean, our nominal β=0.10 is effectively
   β=0.10/N ≈ β=6e-4 (off by ~100-160×); whole sweep was in wrong regime.
2. **Rollout doesn't preserve 1s logged history** (Audit #2 Claim 1) —
   π_ref scores a partly π_θ-generated "history" context.
3. **Non-controlled agents driven by token 0** (Audit #2 Claim 2) —
   set_state writes all-agent poses; non-controlled rows get token-0
   decoded motion, not log-replay. π_ref sees an unphysical frozen scene.
4. **Valid coverage unmeasured** (Audit #3 Concern 2) — effective KL may
   be `β × valid_fraction` rather than `β`.
5. **`alpha` dead in `_ppo_update`/`_collect`** (Audit #2 Claim 3) —
   blocks any Variant-5 (KL + r_inf + LLH) test.

---

## Audit #1 — Test 22 general review

Reviewer's high-level conclusion ("substantially correct… not due to a
code bug") is upheld; some supporting claims hold, some are overstated,
one is empirically contradicted, and one labelled "critical bug" is not
in fact a bug.

## ✅ Claims that are TRUE (confirmed in code)

| # | Claim | Evidence |
|---|---|---|
| 1 (config) | `α=0`, `w_goal=0` in all runs | `train_spacer.py:487` `run(... alpha=0.0 ...)`, `train_spacer.py:70` `goal_achieved_weight=0.0` |
| 6 | `dynamics_model="state"`, `collision_behavior="stop"` | `train_spacer.py:64` |
| 7 | `roadgraph_top_k=120` matches paper §A.3 | `train_spacer.py:77` |
| 8 | `n_samp` counts **only controlled agents** (cmask filter) | `train_spacer.py:184` `x = obs[cmask]` — paper may include more |
| 9 | `ent_coef = 1e-4` (Table A3) | `train_spacer.py:255` `PPO_ENT_COEF = 1e-4` |
| 10 | `torch.max` for clipped surrogate, `norm_adv=True` | confirmed in `_ppo_update` |
| 11 | `align_agents` pre-allocates `[max_id+2]` lookup | `anchor.py:39` — in our runs ids are < 10k so memory fine |

## ⚠️ Claim 3 — the "critical bug" claim is OVERSTATED

**Factually true part:** `tag["gt_idx"]` in `score_ref` comes from
`tp(b)` where `b` is built from `s_live_per_w` (the rolled-out scene). So
`gt_idx` is the tokenizer **re-encoding** the rolled-out poses, *not* the
π_θ-chosen token captured directly during the rollout.

**But the "bug" characterization is wrong**, for two independent reasons:

1. **Round-trip identity.** π_θ outputs token `a`; `decode_token_sequence`
   looks `a` up in `token_traj` → produces a delta-pose; the env executes
   that pose; `tp` then re-tokenizes by argmin over the same `token_traj`
   lookup. **If the lookup is the same table used by both sides — and it
   is, both come from `tp._get_agent_shape_and_token_traj` via
   `policy._ttraj` — the round-trip is exact.** So `gt_idx[t] == a_θ[t]`
   for controlled agents, modulo float precision. r_h is therefore
   actually `log π_ref(π_θ's action)`, exactly as Eq. 3 demands.

2. **The user concedes the training impact themselves:** `α=0` ⇒ `r_h`
   is logged only, never enters the loss. Even if the round-trip *weren't*
   exact, training would be unaffected.

So this isn't a bug — it's an indirection that resolves to the correct
quantity. The closed-form KL (Eq. 5) is genuinely *independent* of
`gt_idx` (it's a distribution-distribution divergence over all 2048
tokens) — the user got that part right.

## ⚠️ Claim 1 — interpretation is OVERSTATED (the *facts* are right)

The framing "no incentive to make progress" treats `r_inf` as the *only*
training signal. **It overlooks the KL anchor**: the policy is being
pulled toward π_ref, and **π_ref does drive to goals** (ref-arm eval:
goal_rate 0.29, minADE 4.2 m). So the KL term *is* a distributional
progress signal — it's not "stay still and avoid penalties with zero
reach toward goals."

The honest version of the claim: at our compute scale, the KL anchor
*isn't strong enough* to transfer π_ref's goal-reaching behaviour into
π_θ. Adding `w_goal>0` is a **paper-deviation** (becomes Variant 3
"Goal + KL", not Variant 4). It might help empirically, but it's no
longer the paper's claimed-best variant.

The paper's own claim is that V4 works *at their scale* — and that may
itself be debatable, but we haven't disproved it; we've only failed to
reproduce it at 1/160th the env-step budget.

## ❌ Claim 4 — β recommendation contradicts our data

The reviewer suggests "try β=1.0 or higher because KL is too weak."
**We already did this in Test 21.** β=1.00 was the **worst** trained run
in our entire project:

| Metric | β=0.10 | β=1.00 | Δ |
|---|---|---|---|
| goal_rate | 0.024 | **0.008** | 3× worse |
| minADE | 27.4 m | **29.6 m** | +2.2 m worse |
| off-road | 0.039 | **0.144** | 3.7× worse |

Test 21 ruled β-up-tuning out empirically. Going higher would likely be
worse, not better.

## ⚠️ Claim 5 — vocab argument is plausible but not rigorous

"More tokens → smaller per-token KL contribution → policy can deviate
more in trajectory space while keeping the same token-level KL" is a
*handwave*. KL over 2048 tokens isn't bounded smaller than KL over 200
tokens in any obvious way: a token distribution can still place mass on
a wrong region with arbitrary KL.

What **is** likely true: with 10× more tokens, **each token represents a
smaller motion increment**, so two trajectories that are close in
token-distribution can still differ by ~10× more in motion at the same
per-token KL. **That's** the real concern, not "spread over more
categories." Still, it's a hypothesis — not proven.

`STAGE_PLAN.md` S2.6 already mentions the token-clustering path; we just
haven't tried it.

## ✅ Claim 2 — agent alignment concern is real but verifiable

The concern is legitimate. `align_agents` masks out unmatched ids. If
many controlled agents lack a π_ref counterpart, the KL is averaged over
fewer samples → weaker signal. Easy to instrument with a one-line print
of `amatch.sum() / amatch.numel()`. The fact that we get a meaningful KL
value (0.65 at convergence) suggests *most* agents match, but it's worth
measuring.

## Items the verification leaves NEUTRAL

- **Claim 8 final sentence** ("paper's batch includes all agents, yours
  only controlled"): unverified — we don't know exactly what the paper's
  PPO buffer includes. Plausible difference but not confirmed.
- **Claim 11 memory blowup**: theoretically real, not realised in our
  scenes. Not a current bug.
- **Claim 12** (finite_diff_velocity / Test 4 NLL): the user references
  documentation, not direct verification. Likely correct but unaudited
  here.

## Summary of the audit

| Verdict | Claims |
|---|---|
| Factually correct + diagnostically useful | 1 (facts), 6, 7, 8, 9, 10, 11 |
| Factually correct, interpretation overstated | 1 (framing), 2, 5 |
| Overstated as "bug" — not actually a bug | **3** |
| Contradicted by our own data | **4** (β=1.0 already tried, worst result in project) |

**Bottom line: the code is correct.** The reviewer reached the same
conclusion ("substantially correct… not due to a code bug") even after
the (unfounded) "critical bug" detour. Their actionable recommendations
are mostly the *deviations from paper-Variant-4* that Test 22 was
deliberately designed to avoid.

## Reviewer's recommendations — accept / reject

| Recommendation | Verdict | Reason |
|---|---|---|
| Higher `ent_coef` (1e-4 → 1e-3) | ✅ worth trying | Sound at our small batch; preserves exploration; paper-deviation but defensible |
| Token vocab coarsening (S2.6) | ✅ worth trying | Actually attacks the locked-by-`clsft_E9` vocab-size deviation we documented in Architecture.md |
| Add `w_goal` (e.g. 0.1) | ❌ reject | Leaves Variant 4 → solving a different variant; not paper-faithful |
| Higher β (e.g. 1.0) | ❌ reject | Empirically refuted by Test 21 — β=1.00 was worst run |
| 5 Hz cadence upgrade | ❌ reject (for now) | Test 20 ref arm proved 2 Hz suffices for good driving (teacher minADE 4.2 m at 2 Hz) |
| Run longer (e.g. 2000 iters at W=24) | ⚖️ marginal | Would help but ~3× our current budget; still ~50× short of paper |

## Open instrumentation idea (raised by Claim 2)

Add `amatch.sum() / amatch.numel()` ratio to the `[scale]` line at iter 0,
so we *know* what fraction of cmask agents have π_ref counterparts. If
it's e.g. 60%, the effective KL signal is 40% weaker than the printed KL
suggests. Cheap to add, no algorithm change.

---

## Audit #2 — sharp surgical review (the one that found real bugs)

This review made 5 line-numbered claims. **Three are real bugs**, one is
dead code, one is a diagnostic-noise issue. The first two are
*potentially significant* — they distort the very thing the KL anchor
depends on (the scene context π_ref sees).

### ✅ Claim 1 — **GENUINE BUG**: rollout doesn't preserve logged history

Confirmed in `rollout()` ([train_spacer.py:165-173](spacer/train_spacer.py#L165-L173)):
```python
prev_pos = [torch.tensor(scenes0[w]["pos_xy"][:, 0], dtype=torch.float32) ...]
prev_head = [torch.tensor(scenes0[w]["yaw"][:, 0], dtype=torch.float32) ...]
steps = list(range(SHIFT, NUM_STEPS, SHIFT))   # [5, 10, ..., 90] — 18 steps
```

The rollout **starts from WOMD step 0** with no logged-history substitution.
First π_θ decision advances step 0 → 5 — *inside* the 1 s history window
(steps 0-10). So `buf_pos[w][:, 5..10]` is **π_θ-generated**, not logged.

`REF_STEP_OFFSET=2` does skip the first two π_θ decisions in the
KL/r_h *alignment*, but π_ref still scores **starting at step 10** using a
*context* (steps 0-10) that's been partly π_θ-generated rather than the
logged WOMD history π_ref was trained on. **This systematically biases
π_ref's logits at step 10+.** Real, non-cosmetic.

`NUM_HIST_STEPS=11` already exists in `gpudrive_to_smart.py:41`
("step 10 = current") — the *concept* of a history window is
acknowledged but not enforced in the rollout.

### ✅ Claim 2 — **GENUINE BUG (potentially large impact)**: non-controlled agents driven by token 0

Confirmed at [train_spacer.py:192-204](spacer/train_spacer.py#L192-L204):
```python
tok_w = torch.zeros(Aw, 1, dtype=torch.long)            # ALL agents init token 0
tok_w[cmask[w].cpu()[:Aw]] = tok_split[w].view(-1, 1)   # only controlled overwritten
dp, dh = decode_token_sequence(tok_w, ...)              # decode ALL agents
dpos_per_w.append(dp[:, 0]); dhead_per_w.append(dh[:, 0])
for _ in range(SHIFT):
    set_state(env, dpos_per_w, dhead_per_w)             # writes ALL agent poses
```

And `set_state` at [train_spacer.py:118-134](spacer/train_spacer.py#L118-L134):
```python
for w in range(W):
    Aw = pos_per_w[w].shape[0]            # ALL agents (not just controlled)
    act[w, :Aw, 0] = pos_per_w[w][:, 0].to(DEV)
    ...
```

So **non-controlled agents are driven by decoded token 0 every iteration,
not log-replay.** Token 0 in the SMART vocab is typically "near-zero
motion / stay still". The scene context π_ref sees has ~50 of ~60 agents
*frozen* at their initial positions instead of following their logged
trajectories. **π_ref's logits for the controlled agents are conditioned
on a fundamentally unphysical scene context.**

For comparison, `eval_quick.py`'s `ref_rollout` ([eval_quick.py:306-310](spacer/eval_quick.py#L306-L310))
*does* get this right — uses GT log-replay for non-controlled agents and
only writes π_ref's positions for the agents π_ref controls. **The
training rollout doesn't do that.** A clear, plausibly large impact bug.

### ✅ Claim 3 — **GENUINE**: `alpha` is dead in `_ppo_update` and `_collect`

Confirmed at [train_spacer.py:369-413](spacer/train_spacer.py#L369-L413):
```python
def _ppo_update(policy, opt, flat, alpha, beta):
    ...
    loss = (pg - PPO_ENT_COEF * entropy.mean()
            + PPO_VF_COEF * v_loss + beta * kl_mb)   # no alpha anywhere
```
And `_collect` only computes `adv, ret = _gae(rew, val, ...)` where
`rew = r_task` (no `+ alpha * r_h`).

For Variant 4 (α=0) this has no effect — but it means **Variant 5
(KL + r_inf + LLH) cannot be tested** without code changes. The CLI flag
and docstring are misleading. `r_h` is computed and logged but never
enters the reward used for GAE.

### ✅ Claim 4 — **GENUINE**: `align_agents` dead allocation

Confirmed at [anchor.py:38-49](spacer/anchor.py#L38-L49):
```python
pos = torch.full((int(ref_ids.max().item()) + 2,), -1, ...) if ref_ids.numel() else ...
# ↑ allocated, never read
lut = {}                                    # actual lookup is a Python dict
for j, tid in enumerate(theta_ids.tolist()):
    lut.setdefault(tid, j)
gather = torch.tensor([lut.get(int(r), 0) for r in ref_ids.tolist()], ...)
valid = torch.tensor([int(r) in lut for r in ref_ids.tolist()], ...)
return gather, valid
```

`pos` is allocated but never used. With small ids (our case), it's
harmless. With large ids it could OOM. Cosmetic; should be deleted.

### ✅ Claim 5 — **GENUINE but low-impact**: dropout in rollout's dual forward

Confirmed at [train_spacer.py:184-185](spacer/train_spacer.py#L184-L185):
```python
logits = policy.logits(x)             # forward #1
tok, lp, _, val = policy(x, deterministic=False)   # forward #2
```

Two separate forwards through a policy with `dropout=0.01`, no `.eval()`
called → different dropout masks → slightly different intermediates. The
*logged* logits differ from what `lp` was sampled from.

**Impact**: `oldlp` in PPO ratio comes from the second forward — self-
consistent. The *logged* KL/entropy use the first forward's logits — so
those diagnostic curves are slightly noisier than necessary. **No training
math error**, just dirty diagnostics. Easy fix: use `forward_with_logits`
once.

### Verdict summary — Audit #2

| Claim | Verdict | Impact |
|---|---|---|
| 1. Rollout starts at step 0, no logged history | ✅ **REAL BUG** | **Significant** — π_ref scores a corrupted context |
| 2. Non-controlled agents driven by token-0 | ✅ **REAL BUG** | **Very significant** — π_ref scores a frozen, unphysical scene |
| 3. `alpha` dead in `_ppo_update` / `_collect` | ✅ Real | None for V4; blocks V5 |
| 4. `align_agents` dead allocation | ✅ Real | Dead code; minor OOM risk for large ids |
| 5. Dropout in dual-forward rollout | ✅ Real | Diagnostic noise only |

**Claims 1 + 2 together** are a strong candidate explanation for why
even paper-batch K=46 still degenerated: both distort what π_ref sees,
so the KL anchor pulls toward an off-distribution target.

---

## Audit #3 — strategic review (the one that surfaced KL normalization)

This review made 5 concerns and 2 "interactions". Three concerns are
real, one is a non-issue, and **Interaction B** is the most important
open question across all three audits.

### ✅ Concern 1 — `ref_logits` stored as raw, `log_softmax` recomputed per minibatch

Confirmed: `ref_logits` at [train_spacer.py:310, 331](spacer/train_spacer.py#L310)
stores raw logits; [train_spacer.py:409](spacer/train_spacer.py#L409) does
`lprf = torch.log_softmax(reflog, dim=-1)` per minibatch. **Correct,
precision-safe**, minor recompute cost (~64 log_softmax/iter).

### ✅ Concern 2 — `valid.sum() / valid.numel()` coverage is NOT logged

Confirmed: `ref_valid[T, N]` is used pervasively but the *fraction* of
True entries is never printed. Anywhere from 100% to 30% would look
identical in our logs. The mean-over-valid masking at
[train_spacer.py:341, 410](spacer/train_spacer.py#L341) means the
*effective* KL weight is `β × coverage`. If coverage is 50%, our β-sweep
β∈{0.01,0.10,1.00} actually covered β∈{0.005,0.05,0.5}. Worth
instrumenting before any further runs.

### ⚠️ Concern 3 — **Reviewer had the direction wrong (first 2, not last 2)**

Confirmed at [train_spacer.py:293, 309, 326, 331](spacer/train_spacer.py#L293):
- `T = len(rec["tok"]) = 18` π_θ decisions per rollout.
- `off = REF_STEP_OFFSET = 2`, `Te = min(T_ref, T - off) = 16`.
- `ref_logits[off:off + Te] = rl[:, :Te]` writes to slots **2…17**.

So slots **0 and 1** (the **first** 2 π_θ decisions) have no ref_logits
and no ref_valid → excluded from KL. **NOT the last 2** as the reviewer
claimed. The underlying observation (2 π_θ decisions are unanchored) is
correct; the direction matters because it's *exactly* the early-rollout
decisions during the WOMD history window (steps t=5, t=10) that go
un-scored — which links directly to Audit #2 Claim 1 (those steps
*should* be logged history, but aren't). Audit #2 Claim 1 and Audit #3
Concern 3 are two facets of one cohesive temporal-alignment issue.

### ❌ Concern 4 — `obs_dim` is auto-detected, **no mismatch risk**

Confirmed at [train_spacer.py:518](spacer/train_spacer.py#L518):
```python
obs0 = env.reset()
odim = obs0[env.cont_agent_mask].shape[-1]
policy = TokenPolicy(obs_dim=odim).to(DEV)
```

`obs_dim` is *always* derived from the actual env's observation shape
— including after `roadgraph_top_k=120`. The `TokenPolicy(obs_dim=2984)`
default in the constructor signature is overridden by the auto-detected
value. **No silent shape mismatch possible.** Non-issue.

### ⚠️ `vbd_in_obs` — non-issue

`vbd_in_obs=False` default ([policy_token.py:73](spacer/policy_token.py#L73)),
consistently used. With it False, no VBD bits in the obs. obs_dim
auto-detection (above) handles whatever GPUDrive actually produces. No
mismatch.

### ⚠️ Interaction A — wide-vocab "stand still attractor"

Plausible *hypothesis*, **untested**. We don't log the policy's action-
token distribution. If most of π_θ's tokens are concentrated on a few
near-stationary IDs, the hypothesis is supported. Cheap test: add
`torch.bincount(act)` print at iter 0 and end.

### 🔥 Interaction B — **KL aggregation: mean vs sum** (the critical unresolved one)

Confirmed at [train_spacer.py:410](spacer/train_spacer.py#L410):
```python
kl_mb = (lpth.exp() * (lpth - lprf)).sum(-1)[rv].mean()
```
- `.sum(-1)` collapses 2048-vocab axis → per-(agent, step) KL.
- `[rv].mean()` averages across valid (agent, step) pairs in the minibatch.

So `β·kl_mb` is `β × (1/n_valid) Σ D_KL(agent, step)`. **Per-sample mean.**

This matches standard PPO convention (`pg`, `v_loss`, `entropy.mean()`
are all per-sample means), so the *relative scale* of KL to the other
loss terms is right.

**But the reviewer's question is whether the paper's reference
implementation uses MEAN or SUM over agents.** If the paper sums and we
mean, our β=0.10 has been effectively β=0.10/N (N≈100-160 valid
(agent, step) per minibatch).

| Our nominal β | If paper sums: effective β (× ~1/N=1/130) |
|---|---|
| 0.01 | ≈ 7.7e-5 |
| 0.10 | ≈ 7.7e-4 |
| 1.00 | ≈ 7.7e-3 |

Paper's stated β=0.01 ≈ our β=1.30. This would re-frame Test 21: **β=1.00
being worst on driving but closest to paper's effective regime** would
suggest we'd need β >> 1.00 to actually reach the paper's anchor
strength. *Or* per-sample mean is correct convention and our nominal β
values are right.

**Verdict**: cannot verify without inspecting the paper's actual code
or PufferLib's HR-PPO KL term directly. **The single most important
unresolved question** — could explain almost everything Test 22 didn't.
Cheap to audit by reading `gpudrive/integrations/puffer/ppo.py`.

### Verdict summary — Audit #3

| Claim | Verdict | Impact |
|---|---|---|
| 1. ref_logits storage format | ✅ correct | None |
| 2. `valid` coverage unmeasured | ✅ legit | Possibly real (signal weakness) |
| 3. T=18 vs T_ref=16 (first 2 unanchored) | ✅ confirmed; reviewer had direction wrong | Same root cause as Audit #2 Claim 1 |
| 4. obs_dim mismatch | ❌ non-issue (auto-detected) | None |
| `vbd_in_obs` | ❌ non-issue | None |
| Interaction A (vocab + stand-still attractor) | ⚠️ plausible, untested | Could be real |
| **Interaction B (KL mean vs sum)** | ⚠️ **unknown, potentially huge** | **Audit PufferLib HR-PPO** |

---

## Synthesis across all three audits

What we now collectively know:

| Category | Findings |
|---|---|
| **Real bugs that could affect Test 22's degenerate result** | (A2-1) no logged history in rollout; (A2-2) non-controlled agents on token 0; (A3-B) possible β-normalization mismatch |
| **Lower-impact real code issues** | (A2-3) `alpha` not wired into GAE; (A2-4) `align_agents` dead alloc; (A2-5) dropout dual-forward |
| **Unmeasured, cheap to instrument** | (A3-2) valid-coverage fraction; (A3-A) token distribution |
| **Conclusively non-issues** | (A1-3) `gt_idx` re-tokenization "critical bug" — round-trip exact, α=0 anyway; (A3-4) obs_dim mismatch — auto-detected |
| **Empirically refuted by our data** | (A1-4) higher β recommendation — Test 21 disproved β=1.00 |

### Recommended next moves (ordered by expected leverage / cost ratio)

1. **Audit PufferLib's HR-PPO KL term** for mean vs sum convention. Zero
   compute cost; potentially reframes the entire β picture. If we've
   been silently in the wrong β regime, every β-sweep result needs
   re-interpretation.
2. **Fix Audit #2 Claim 2** (non-controlled agents → log-replay) — copy
   the pattern already correct in `eval_quick.py`'s `ref_rollout`. This
   is a focused ~15-line patch and the bug it fixes is plausibly large.
3. **Fix Audit #2 Claim 1** (use logged history for steps 0-10, start
   rollout at step 10). Resolves Audit #3 Concern 3 too.
4. **Instrument** valid-coverage fraction (Audit #3 #2) and token-
   distribution (Audit #3 Interaction A) — one print each at iter 0.
5. Defer: Audit #2 #3 (`alpha`), #4 (dead alloc), #5 (dropout). Cosmetic
   or out-of-scope-for-Variant-4.
