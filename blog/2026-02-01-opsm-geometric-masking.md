---
layout: post
title: "deepseek off policy sequence masking is a geometric sequence mask"
categories: [RL]
year: 2025
type: blog
---

Geometric Sequence Masking is a technique for importance sampling correction in RL training that addresses a fundamental problem: when your inference engine and training framework produce slightly different distributions, standard sequence-level importance weights become length-biased, systematically penalizing longer generations. Geometric masking fixes this by using the geometric mean of per-token importance ratios, which is length-invariant.

DeepSeek's Off-Policy Sequence Masking (OPSM), introduced in the V3.2 technical report, turns out to be mathematically equivalent to a one-sided geometric sequence mask. The "KL threshold" in OPSM is just the negative log of a geometric mean importance ratio. This equivalence means you can implement OPSM by extending an existing geometric masking setup, and it provides a cleaner way to understand what OPSM is actually doing: it's handling both training-inference mismatch and policy staleness through a single, length-invariant masking criterion.

## background: the ppo ratio

PPO's clipped surrogate objective contains a ratio between policies:

$$
J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right)
$$

This ratio emerges from importance sampling. The policy gradient requires expectations over trajectories from the current policy $\pi_\theta$, but we want to take multiple gradient steps on a batch collected from a fixed policy $\pi_{\theta_{\text{old}}}$. The importance weight $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ corrects for this distribution mismatch.

This correction enables multiple gradient steps per batch, which is crucial for efficiency. Generating rollouts is expensive—we want to squeeze as much learning as possible from each generation pass.

## the three policy formulation

Standard PPO conflates two distinct roles into a single "old" policy. The batch-size invariant PPO paper (Schulman et al.) decouples these into separate policies:

**Behavior policy** ($\mu$): The policy that actually generated the data. It appears in the importance sampling ratio that corrects for off-policy data:

$$
\frac{\pi_{\theta}(a|s)}{\mu(a|s)}
$$

**Proximal policy** ($\pi_{\text{old}}$): A recent policy used as an anchor for the trust region. It appears in the clipping term that prevents the policy from changing too drastically:

$$
\text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\text{old}}(a|s)}, 1-\epsilon, 1+\epsilon\right)
$$

In standard PPO, $\mu = \pi_{\text{old}}$—the same policy serves both roles. The decoupled formulation separates them:

$$
J(\theta)_{\text{decoupled}} = \frac{\pi_{\text{old}}(a | s)}{\mu(a | s)} \min\left(\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right)
$$

The leading term $\frac{\pi_{\text{old}}}{\mu}$ handles off-policy correction. The ratio inside the min/clip handles trust region enforcement. These are separate concerns.

The original GRPO formulation used standard PPO's clipped surrogate objective without this explicit separation—it assumed $\mu = \pi_{\text{old}}$ like vanilla PPO. The three policy formulation became relevant for LLM training through two developments: AReaL introduced it to handle asynchronous RL where multiple policy versions generate training data, and a [blog post from Fengyao](https://fengyao.notion.site/off-policy-rl) highlighted it specifically for training-inference mismatch in disaggregated systems.

## why this matters for llm training

Modern RL training for LLMs uses disaggregated systems: a high-throughput inference engine (vLLM, SGLang) generates rollouts, while a separate training framework performs gradient updates. This separation exists because inference engines are optimized for generation throughput, training frameworks for gradient computation throughput. The problem is that these systems can produce different probability distributions even when running identical model weights.

This discrepancy was noted in the [MiniMax M1 paper](https://arxiv.org/abs/2506.13585), but it wasn't until [Fengyao's blog post](https://fengyao.notion.site/off-policy-rl) that someone systematically studied the problem and proposed corrections. The [Ant Group paper](https://arxiv.org/abs/2510.18855) expanded on this with additional masking strategies. The core insight from this line of work: in disaggregated RL, we're not just dealing with policy staleness from multiple gradient steps—we're also dealing with a fundamental mismatch between the sampling distribution and the training distribution.

This means two sources of off-policyness:

**Training-inference mismatch**: The sampling distribution from the inference engine ($\mu$) differs from what the training model ($\pi_{\text{old}}$) would have produced due to quantization, different attention implementations, numerical precision, or softmax implementations.

**Policy staleness**: RL systems generate large rollout batches, then split them into mini-batches for multiple gradient steps. By the time you're training on the last mini-batch, the policy $\pi$ has moved from $\pi_{\text{old}}$.

The three policy formulation gives us language to discuss these separately. We track three sets of log-probs:

| Symbol | Variable | Description |
|--------|----------|-------------|
| $\pi$ | `per_token_logps` | Current policy being optimized |
| $\pi_{\text{old}}$ | `old_per_token_logps` | Training model at rollout time (proximal policy) |
| $\mu$ | `sampling_per_token_logps` | Inference engine at rollout time (behavior policy) |

The ratio $\frac{\pi_{\text{old}}}{\mu}$ captures training-inference mismatch—a correction the original PPO and GRPO objectives don't account for. The ratio $\frac{\pi}{\pi_{\text{old}}}$ captures policy staleness, which is what the standard PPO clipping mechanism constrains. The former requires explicit importance sampling correction beyond what PPO provides.

## importance sampling correction

So how do we actually correct for training-inference mismatch? The standard tool is importance sampling. If we have samples from a behavior policy $\mu$ but want to estimate expectations under a target policy $\pi$, we reweight each sample by the ratio of how likely it is under the target versus the behavior:

$$
\mathbb{E}_{y \sim \pi}[f(y)] = \mathbb{E}_{y \sim \mu}\left[\frac{\pi(y)}{\mu(y)} f(y)\right]
$$

For autoregressive generation, computing this ratio requires a product over all tokens. Each token's probability depends on the previous tokens, so the sequence-level ratio factors as:

$$
\frac{\pi(y|x)}{\mu(y|x)} = \prod_{t=0}^{T-1} \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

We can write this more compactly by defining $\rho_t = \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}$ for the per-token ratio, giving us $\rho_{\text{seq}} = \prod_t \rho_t$ for the sequence.

For training-inference mismatch specifically, we want to correct between the training model $\pi_{\text{old}}$ and the inference engine $\mu$:

$$
\rho_t^{\text{TI}} = \frac{\pi_{\text{old}}(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

In practice, unbounded importance weights can destabilize training—a single sample with extreme weight dominates the gradient. Two approaches bound these ratios:

**Truncated Importance Sampling (TIS)** clips ratios to a range, keeping the gradient signal but limiting its magnitude:

$$
\rho \leftarrow \text{clip}(\rho, C_{\min}, C_{\max})
$$

**Masked Importance Sampling (MIS)** discards samples with extreme ratios entirely, enforcing a hard trust region:

$$
\rho \leftarrow \begin{cases} \rho & \text{if } C_{\min} \leq \rho \leq C_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

Both can be applied per-token (checking each $\rho_t$) or per-sequence (checking the product $\rho_{\text{seq}}$). In code, a sequence-level MIS check looks like:

```python
# Variable mapping:
#   old_per_token_logps = π_old (training model)
#   sampling_per_token_logps = μ (inference engine)

# Per-token log ratio: log(π_old / μ)
per_token_log_ratio = (old_per_token_logps - sampling_per_token_logps) * mask

# Sequence-level: sum of logs = log of product
seq_log_ratio = per_token_log_ratio.sum(dim=-1)
seq_ratio = torch.exp(seq_log_ratio)

# MIS: mask sequences outside bounds
mis_mask = (seq_ratio >= C_min) & (seq_ratio <= C_max)
```

The sequence-level approach is simpler but has a problem we'll see next.

## the length bias problem

The sequence-level approach has a fundamental issue: the importance ratio $\rho_{\text{seq}} = \prod_t \rho_t$ grows or shrinks exponentially with sequence length.

Consider a modest per-token drift of 0.1%—$\rho_t \approx 1.001$ on average. Over 2000 tokens:

$$
\rho_{\text{seq}} = (1.001)^{2000} \approx 7.39
$$

Even with minimal per-token divergence, long sequences get systematically rejected or heavily down-weighted. The effective trust region becomes sequence-length dependent: a bound like $C_{\max} = 2$ might accept most 100-token sequences but reject almost all 2000-token sequences. This creates a bias toward shorter generations regardless of their quality.

## geometric sequence masking

The fix is to use a length-invariant measure of divergence. Instead of the product of per-token ratios, Geometric Sequence Masking uses their geometric mean:

$$
\rho_{\text{geo}} = \left( \prod_{t=0}^{T-1} \rho_t \right)^{1/T}
$$

In log-space, the geometric mean becomes an arithmetic mean, which is easier to compute:

$$
\log \rho_{\text{geo}} = \frac{1}{T} \sum_{t=0}^{T-1} \log \rho_t = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t|x, y_{<t})}{\mu(y_t|x, y_{<t})}
$$

This is the average per-token log-likelihood ratio. Crucially, it doesn't scale with $T$—a 100-token sequence and a 2000-token sequence with the same average per-token divergence will have the same $\rho_{\text{geo}}$.

The geometric sequence mask then applies bounds on this length-invariant ratio:

$$
g_{\text{geo-mask}} = \mathbf{1}\left[ C_{\min} \leq \rho_{\text{geo}} \leq C_{\max} \right]
$$

In code, we can extend the MIS calculation from before. The only change is dividing by sequence length before checking bounds:

```python
# Same setup as MIS
per_token_log_ratio = (old_per_token_logps - sampling_per_token_logps) * mask
seq_log_ratio = per_token_log_ratio.sum(dim=-1)

# Geometric mean: divide by sequence length
seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
geo_mean_log_ratio = seq_log_ratio / seq_lengths  # <-- this is the key change
geo_ratio = torch.exp(geo_mean_log_ratio)

# Same masking logic, now length-invariant
geo_mask = (geo_ratio >= C_min) & (geo_ratio <= C_max)
```

So far we've covered importance sampling correction for training-inference mismatch—specifically, how to compute the ratio $\frac{\pi_{\text{old}}}{\mu}$ and how geometric masking makes this length-invariant. But this only addresses one source of off-policyness. Policy staleness—the drift between $\pi$ and $\pi_{\text{old}}$ during multiple gradient steps—is a separate problem.

DeepSeek's OPSM tackles both at once, using a different approach than the TIS/MIS methods above.

## deepseek opsm

[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) introduces Off-Policy Sequence Masking. Like the MIS we saw earlier, it's a sequence-level mask that discards samples based on a divergence threshold. But OPSM has two key differences: it measures divergence between the current policy $\pi$ and the sampling distribution (not between $\pi_{\text{old}}$ and $\mu$), and it only applies to negative-advantage samples.

The masking rule:

$$
M_{i,t} = \begin{cases}
0 & \text{if } \hat{A}_{i,t} < 0 \quad \text{and} \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_{\text{old}}(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})} > \delta \\
1 & \text{otherwise}
\end{cases}
$$

A sequence is kept if either: (1) it has non-negative advantage, or (2) its average log-probability ratio is within threshold $\delta$. The rationale from the paper:

> Models benefit the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process.

The one-sided nature makes sense: for positive-advantage samples, we want to reinforce the behavior regardless of how much the policy has drifted. For negative-advantage samples, learning from highly off-policy data can be counterproductive—the policy has already moved away from these samples, suggesting they're no longer representative of what the current policy would produce.

A crucial detail from DeepSeek about the notation:

> Note that $\pi_{\text{old}}$ here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above.

In our three-policy notation, DeepSeek's $\pi_{\text{old}}$ is our $\mu$—the inference engine's distribution. So OPSM computes the divergence between the current policy $\pi$ and the inference distribution $\mu$, which captures both training-inference mismatch and policy staleness in a single ratio. This is different from the TIS/MIS approaches that only correct for training-inference mismatch between $\pi_{\text{old}}$ and $\mu$.

## opsm is geometric sequence masking

Look at OPSM's threshold term again: $\frac{1}{T}\sum_t \log \frac{\mu}{\pi}$. This is the average per-token log-probability ratio—exactly the log of a geometric mean, just with the policies in a different order than we used for training-inference correction. If we define the geometric mean ratio as $\rho_{\text{geo}} = \exp\left(\frac{1}{T}\sum_t \log \frac{\pi}{\mu}\right)$, then OPSM's threshold term is simply $-\log \rho_{\text{geo}}$.

This means OPSM's condition "$\frac{1}{T}\sum_t \log \frac{\mu}{\pi} > \delta$" is equivalent to:

$$
-\log \rho_{\text{geo}} > \delta \quad \Leftrightarrow \quad \rho_{\text{geo}} < e^{-\delta}
$$

OPSM is a one-sided geometric sequence mask. It drops sequences where the current policy assigns significantly lower probability than the sampling distribution—where $\rho_{\text{geo}}$ falls below the threshold $e^{-\delta}$.

| Aspect | Geometric Sequence Mask | DeepSeek OPSM |
|--------|------------------------|---------------|
| Direction | Two-sided ($C_{\min} \leq \rho_{\text{geo}} \leq C_{\max}$) | One-sided ($\rho_{\text{geo}} \geq e^{-\delta}$) |
| Advantage condition | None | Only applies when $\hat{A} < 0$ |
| Ratio | $\frac{\pi_{\text{old}}}{\mu}$ (training-inference only) | $\frac{\pi}{\mu}$ (both sources) |

The key difference is which ratio OPSM uses. We can factor $\frac{\pi}{\mu}$ to see what it captures:

$$
\frac{\pi}{\mu} = \frac{\pi_{\text{old}}}{\mu} \times \frac{\pi}{\pi_{\text{old}}}
$$

The first term is the training-inference mismatch we've been discussing. The second is policy staleness—how much $\pi$ has drifted from $\pi_{\text{old}}$ during gradient updates. OPSM's single ratio captures both.

Taking logs and averaging over the sequence, we get:

$$
\underbrace{\frac{1}{T} \sum_{t} \log \frac{\pi}{\mu}}_{\log \rho_{\text{geo}}} = \underbrace{\frac{1}{T} \sum_{t} \log \frac{\pi_{\text{old}}}{\mu}}_{\text{training-inference geo-mean}} + \underbrace{\frac{1}{T} \sum_{t} \log \frac{\pi}{\pi_{\text{old}}}}_{\text{staleness geo-mean}}
$$

Notice that the first term on the right is exactly the `geo_mean_log_ratio` we computed for training-inference correction earlier. This is a useful decomposition for implementation: the training-inference term only needs to be computed once per rollout (it doesn't depend on $\pi$), while the staleness term changes with each gradient step as $\pi$ updates.

If you already have geometric masking for training-inference mismatch, you can extend it to full OPSM by adding the staleness term:

```python
# geo_mean_log_ratio from earlier: log geometric mean of (π_old / μ)
# This is computed once per rollout
ti_geo_mean = geo_mean_log_ratio  # from training-inference geo-mask

# Staleness term: log geometric mean of (π / π_old)
# Computed each gradient step as π updates
staleness_log_ratio = (per_token_logps - old_per_token_logps) * mask
staleness_geo_mean = staleness_log_ratio.sum(dim=-1) / seq_lengths

# OPSM's full ratio: log geometric mean of (π / μ)
opsm_geo_mean = ti_geo_mean + staleness_geo_mean

# OPSM mask: keep if positive advantage OR low divergence
# Note: OPSM uses -log(ρ_geo) > δ, which is equivalent to log(ρ_geo) < -δ
is_neg_adv = advantages < 0
is_high_divergence = opsm_geo_mean < -delta
opsm_mask = ~(is_neg_adv & is_high_divergence)
```

The payoff: if you've already implemented geometric masking for training-inference correction, OPSM is just adding one more term and a conditional on advantage sign.

## summary

The mathematical relationships:

1. **Sequence IS** uses $\prod_t \rho_t$—length-biased
2. **Geometric masking** uses $(\prod_t \rho_t)^{1/T}$—length-invariant
3. **OPSM KL term** equals $-\log \rho_{\text{geo}}$
4. **OPSM** = one-sided geometric mask on $\frac{\pi}{\mu}$, conditioned on negative advantages
5. **Factored**: $\frac{\pi}{\mu} = \frac{\pi_{\text{old}}}{\mu} \times \frac{\pi}{\pi_{\text{old}}}$

The insight that OPSM is a geometric sequence mask connects it to a principled framework for length-invariant importance sampling and clarifies what the method optimizes.

## references

Everything I've come to know about Geometric Sequence Masking, and most of the stuff I've come to understand about importance sampling, is from Yingru LI, and his posts:

1) https://richardli.xyz/post/rl-collapse-part3/
2) https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

