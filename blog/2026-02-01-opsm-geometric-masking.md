---
layout: post
title: "DeepSeek's Off-Policy Sequence Masking is Geometric Sequence Masking"
categories: [RL]
year: 2025
type: blog
---

DeepSeek's Off-Policy Sequence Masking (OPSM), introduced in their V3.2 technical report, is mathematically equivalent to a one-sided geometric sequence mask. This equivalence provides two practical benefits: you can implement OPSM by reusing existing geometric masking code, and you gain a principled framework for understanding what OPSM optimizes. The "KL threshold" in OPSM is simply the log of a geometric ratio threshold.

This post develops the theory that leads to this equivalence, starting from the problem that geometric masking was designed to solve.

## The length bias problem in importance sampling

Reinforcement learning for language models faces a fundamental tension: generating rollouts is expensive, so we want to reuse data across multiple gradient steps. But as training progresses, the policy drifts from the distribution that generated the data. Importance sampling corrects for this mismatch.

For autoregressive generation, the sequence-level importance ratio decomposes as a product of per-token ratios:

$$
\frac{\pi(y|x)}{\mu(y|x)} = \prod_{t=0}^{T-1} \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

where $\pi$ is the target policy and $\mu$ is the behavior policy that generated the data. This product structure creates a problem: the ratio grows or shrinks exponentially with sequence length.

Consider a modest per-token drift where $\rho_t \approx 1.001$ on average. Over 2000 tokens:

$$
\rho_{\text{seq}} = (1.001)^{2000} \approx 7.39
$$

Even minimal per-token divergence causes long sequences to be systematically rejected or heavily down-weighted. The effective trust region shrinks dramatically with length, and the model may learn to generate shorter sequences simply to avoid penalties.

**Geometric sequence masking** solves this by using the geometric mean of importance ratios instead of their product:

$$
\rho_{\text{geo}} = \left( \prod_{t=0}^{T-1} \rho_t \right)^{1/T}
$$

In log-space, this becomes the arithmetic mean of log-ratios:

$$
\log \rho_{\text{geo}} = \frac{1}{T} \sum_{t=0}^{T-1} \log \rho_t
$$

This quantity represents the *average per-token log-likelihood ratio*. The division by $T$ makes it length-invariant: a 100-token sequence and a 2000-token sequence are judged by the same per-token standard.

## Sources of off-policyness in LLM training

Before diving into the mechanics of importance sampling correction, we need to understand *where* off-policyness comes from. Modern RL systems for LLMs use disaggregated architectures: a high-throughput inference engine (vLLM, SGLang) generates rollouts, while a separate training framework performs gradient updates. This separation exists because the two workloads have different optimization targets—generation throughput versus gradient computation throughput.

This architecture introduces two distinct sources of distribution mismatch:

**Training-inference mismatch**: The sampling distribution from the inference engine ($\mu$) may differ from the training model ($\pi_{\text{old}}$) even when they share identical weights. Differences in quantization, attention implementations, numerical precision, or softmax approximations can cause the actual sampled distribution to diverge from what the training framework would compute.

**Policy staleness**: RL systems generate large rollout batches, then split them into mini-batches for multiple gradient steps. By the time you're training on the final mini-batch, the policy $\pi$ has changed significantly from the $\pi_{\text{old}}$ that existed at rollout time. The data is stale.

To discuss these separately, we adopt a three-policy formulation from the batch-size invariant PPO work (Schulman et al.). This decouples the standard "old policy" into two roles:

| Symbol | Role | Description |
|--------|------|-------------|
| $\mu$ | Behavior policy | The inference engine's actual sampling distribution |
| $\pi_{\text{old}}$ | Proximal policy | Training model at rollout time, anchor for trust region |
| $\pi$ | Current policy | Policy being optimized |

The ratio $\frac{\pi_{\text{old}}}{\mu}$ captures training-inference mismatch—it's constant for a given rollout. The ratio $\frac{\pi}{\pi_{\text{old}}}$ captures policy staleness—it changes with each gradient step. Standard PPO clipping operates on the latter. The former requires explicit importance sampling correction.

## Importance sampling correction methods

Given samples from behavior policy $\mu$, we estimate expectations under target policy $\pi$ using importance weights:

$$
\mathbb{E}_{y \sim \pi}[f(y)] = \mathbb{E}_{y \sim \mu}\left[\frac{\pi(y)}{\mu(y)} f(y)\right]
$$

For RL, $f(y)$ is typically the policy gradient term. The importance weight corrects for the distribution shift, but unbounded weights introduce high variance. Two approaches handle this:

**Truncated Importance Sampling (TIS)** clips the ratio to a bounded range:
$$
\rho \leftarrow \text{clip}(\rho, C_{\min}, C_{\max})
$$

**Masked Importance Sampling (MIS)** completely discards samples outside the range:
$$
\rho \leftarrow \begin{cases} \rho & \text{if } C_{\min} \leq \rho \leq C_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

Both can operate at token-level or sequence-level. Here's how these look in code for sequence-level correction of training-inference mismatch:

```python
# Inputs:
#   old_per_token_logps: log π_old(y_t | x, y_{<t}) - training model at rollout
#   sampling_per_token_logps: log μ(y_t | x, y_{<t}) - inference engine
#   mask: attention mask for valid tokens

# Per-token log importance ratio: log(π_old / μ)
log_ratio = (old_per_token_logps - sampling_per_token_logps) * mask

# Sequence-level: sum of log-ratios = log of product
seq_log_ratio = log_ratio.sum(dim=-1)
seq_ratio = torch.exp(seq_log_ratio)

# Truncated IS: clip the ratio
tis_ratio = torch.clamp(seq_ratio, min=C_min, max=C_max)

# Masked IS: zero out samples outside the trust region
mis_mask = (seq_ratio >= C_min) & (seq_ratio <= C_max)
mis_ratio = torch.where(mis_mask, seq_ratio, torch.zeros_like(seq_ratio))
```

The problem with sequence-level ratios is the length bias discussed earlier. For long sequences, even small per-token drift compounds into extreme sequence ratios, causing systematic rejection.

## Geometric sequence masking in practice

Geometric masking replaces the sequence-level product with the geometric mean:

```python
# Inputs same as above

# Per-token log importance ratio
log_ratio = (old_per_token_logps - sampling_per_token_logps) * mask

# Sequence-level sum (same as before)
seq_log_ratio = log_ratio.sum(dim=-1)

# Key change: normalize by sequence length
seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
geo_log_ratio = seq_log_ratio / seq_lengths  # Average per-token log-ratio

# Convert to ratio space
geo_ratio = torch.exp(geo_log_ratio)

# Apply mask based on geometric ratio
geo_mask = (geo_ratio >= C_min) & (geo_ratio <= C_max)
```

The only change is dividing by sequence length. This transforms the length-biased sequence ratio into a length-invariant geometric mean. Typical thresholds are $C_{\min} = 0.5$, $C_{\max} = 2.0$, corresponding to $|\log \rho_{\text{geo}}| \leq \log 2 \approx 0.69$.

## DeepSeek OPSM

[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) introduces Off-Policy Sequence Masking to stabilize RL training. The method addresses a specific failure mode: highly off-policy negative samples can destabilize optimization. From the paper:

> Models benefit the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process.

The masking rule is:

$$
M_{i,t} = \begin{cases}
0 & \text{if } \hat{A}_{i,t} < 0 \quad \text{and} \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_{\text{old}}(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})} > \delta \\
1 & \text{otherwise}
\end{cases}
$$

A sequence is masked (zeroed) if *both* conditions hold: (1) it has negative advantage, and (2) its average KL divergence exceeds threshold $\delta$.

The one-sided nature is intentional. Negative-advantage samples that have drifted far from the current policy are unreliable learning signals—they represent mistakes the model wouldn't make anymore. Positive-advantage samples, even if off-policy, still provide useful reward signal.

A crucial implementation detail from DeepSeek:

> Note that $\pi_{\text{old}}$ here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above.

DeepSeek uses the inference engine's log-probs directly. In our three-policy notation, their $\pi_{\text{old}}$ is our $\mu$. The ratio $\frac{\mu}{\pi}$ therefore captures *both* training-inference mismatch and policy staleness in a single term.

## OPSM is geometric sequence masking

The OPSM formulation may look like a KL divergence threshold, but it's algebraically equivalent to a geometric sequence mask. Recognizing this equivalence lets us: (1) implement OPSM using existing geometric masking code, and (2) understand OPSM within a principled framework for length-invariant importance sampling.

The OPSM condition uses an average log-ratio:

$$
\frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\mu(y_t | x, y_{<t})}{\pi(y_t | x, y_{<t})} > \delta
$$

Let's connect this to the geometric importance ratio $\rho_{\text{geo}} = \left(\prod_t \frac{\pi}{\mu}\right)^{1/T}$. Taking the log:

$$
\log \rho_{\text{geo}} = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

Notice the OPSM KL term is exactly the *negative* of this:

$$
\underbrace{\frac{1}{T} \sum_t \log \frac{\mu}{\pi}}_{\text{OPSM KL term}} = -\underbrace{\frac{1}{T} \sum_t \log \frac{\pi}{\mu}}_{\log \rho_{\text{geo}}}
$$

The OPSM condition $\frac{1}{T} \sum_t \log \frac{\mu}{\pi} > \delta$ therefore becomes:

$$
-\log \rho_{\text{geo}} > \delta \quad \Leftrightarrow \quad \rho_{\text{geo}} < e^{-\delta}
$$

**OPSM is a one-sided geometric sequence mask** that drops sequences where the current policy assigns significantly *lower* probability than the sampling distribution. The threshold $\delta$ maps directly to a geometric ratio threshold $e^{-\delta}$.

| Aspect | Two-sided Geometric Mask | DeepSeek OPSM |
|--------|-------------------------|---------------|
| Bounds | $C_{\min} \leq \rho_{\text{geo}} \leq C_{\max}$ | $\rho_{\text{geo}} \geq e^{-\delta}$ (lower bound only) |
| Advantage condition | None | Only applies when $\hat{A} < 0$ |
| Ratio direction | $\frac{\pi_{\text{old}}}{\mu}$ or $\frac{\pi}{\mu}$ | $\frac{\pi}{\mu}$ |

## Decomposing the OPSM ratio

Since OPSM uses $\frac{\pi}{\mu}$ rather than $\frac{\pi_{\text{old}}}{\mu}$, it captures both sources of off-policyness. We can factor this to see both contributions:

$$
\frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})} = \underbrace{\frac{\pi_{\text{old}}(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}}_{\text{training-inference}} \times \underbrace{\frac{\pi(y_t | x, y_{<t})}{\pi_{\text{old}}(y_t | x, y_{<t})}}_{\text{staleness}}
$$

Taking logs and averaging over the sequence:

$$
\log \rho_{\text{geo}}^{\text{total}} = \underbrace{\frac{1}{T} \sum_t \log \frac{\pi_{\text{old}}}{\mu}}_{\text{training-inference (constant per rollout)}} + \underbrace{\frac{1}{T} \sum_t \log \frac{\pi}{\pi_{\text{old}}}}_{\text{staleness (changes each step)}}
$$

The first term is computed once when the rollout is generated. The second term must be recomputed at each gradient step as $\pi$ updates. This decomposition is useful for debugging and ablations:

```python
# Inputs:
#   per_token_logps: log π(y_t | x, y_{<t}) - current policy
#   old_per_token_logps: log π_old(y_t | x, y_{<t}) - training model at rollout
#   sampling_per_token_logps: log μ(y_t | x, y_{<t}) - inference engine
#   advantages: advantage estimates per sequence
#   delta: OPSM threshold

seq_lengths = mask.sum(dim=-1).clamp(min=1.0)

# --- Computed once per rollout ---
# Training-inference geometric ratio: (π_old / μ)^(1/T)
ti_log_ratio = ((old_per_token_logps - sampling_per_token_logps) * mask).sum(dim=-1)
ti_geo_log = ti_log_ratio / seq_lengths

# --- Computed at each gradient step ---
# Staleness geometric ratio: (π / π_old)^(1/T)
staleness_log_ratio = ((per_token_logps - old_per_token_logps) * mask).sum(dim=-1)
staleness_geo_log = staleness_log_ratio / seq_lengths

# Total geometric log-ratio: log(π/μ)^(1/T) = log(π_old/μ)^(1/T) + log(π/π_old)^(1/T)
total_geo_log = ti_geo_log + staleness_geo_log

# OPSM condition: mask if negative advantage AND geo_ratio below threshold
# KL term in OPSM = -total_geo_log, condition is KL > delta
# Equivalent: total_geo_log < -delta, i.e., geo_ratio < exp(-delta)
is_neg_advantage = advantages < 0
is_off_policy = total_geo_log < -delta  # equivalent to geo_ratio < exp(-delta)
opsm_mask = ~(is_neg_advantage & is_off_policy)

# Apply mask to loss
masked_loss = loss * opsm_mask.float()
```

The practical benefit of this decomposition: `ti_geo_log` can be cached once per rollout since it doesn't depend on the current policy. Only `staleness_geo_log` needs recomputation at each gradient step.

## Summary

The key insight is that OPSM's "average KL threshold" is algebraically equivalent to a geometric importance ratio threshold:

1. **Sequence-level IS** uses $\prod_t \rho_t$—exponentially length-biased
2. **Geometric masking** uses $(\prod_t \rho_t)^{1/T}$—length-invariant
3. **OPSM's KL term** equals $-\log \rho_{\text{geo}}$
4. **OPSM** = one-sided geometric mask on $\frac{\pi}{\mu}$, conditioned on negative advantages
5. **The ratio factors** as $\frac{\pi}{\mu} = \frac{\pi_{\text{old}}}{\mu} \times \frac{\pi}{\pi_{\text{old}}}$, separating training-inference mismatch from staleness

This connection places OPSM within the broader framework of length-invariant importance sampling correction. If you're already computing geometric sequence masks for training-inference mismatch, extending to OPSM requires only: (1) using $\pi$ instead of $\pi_{\text{old}}$ in the numerator, and (2) conditioning the mask on negative advantages.

## References

The geometric sequence masking framework and much of the importance sampling theory here draws from Yingru Li's work:

1. [RL Collapse Part 3](https://richardli.xyz/post/rl-collapse-part3/)
2. [Demystifying RL Collapse from Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
