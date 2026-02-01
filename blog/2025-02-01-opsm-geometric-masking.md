---
layout: post
title: "deepseek off policy sequence masking is a geometric sequence mask"
categories: [RL]
year: 2025
type: blog
---

DeepSeek's Off-Policy Sequence Masking (OPSM), introduced in the V3.2 technical report, is mathematically equivalent to a one-sided geometric sequence mask. This might sound like a minor notational curiosity, but it's not. Understanding this equivalence clarifies what OPSM is actually doing, connects it to a broader class of length-invariant importance sampling corrections, and provides a cleaner mental model for implementing and extending these techniques. The goal of this post is to walk through the theory that leads to this insight.

## the off-policy problem

Modern RL training for LLMs typically uses a disaggregated setup: a high-throughput inference engine (vLLM, SGLang) generates rollouts, while a separate training framework (PyTorch, DeepSpeed) performs gradient updates. This separation is driven by efficiency—inference engines are optimized for generation throughput, training frameworks for gradient computation. But it introduces a subtle problem.

The policy that generates the data is not quite the same as the policy being trained. This shows up in two ways.

First, there's **training-inference mismatch**. The sampling distribution from the inference engine may differ from the training model due to quantization, different attention implementations, numerical precision, or softmax implementations. Even with identical weights, the distributions can diverge.

Second, there's **policy staleness**. RL systems generate large rollout batches, then split them into mini-batches for multiple gradient steps. By the time you're training on the last mini-batch, the policy has changed from several updates. The data is stale.

Both of these create off-policy data: we're training on samples from one distribution while trying to optimize another.

## importance sampling correction

The standard fix for distribution mismatch is importance sampling. If we have samples from a behavior policy $\mu$ but want to estimate expectations under a target policy $\pi$, we reweight:

$$
\mathbb{E}_{y \sim \pi}[f(y)] = \mathbb{E}_{y \sim \mu}\left[\frac{\pi(y)}{\mu(y)} f(y)\right]
$$

For autoregressive generation, the sequence-level importance ratio is a product of per-token ratios:

$$
\frac{\pi(y|x)}{\mu(y|x)} = \prod_{t=0}^{T-1} \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

We can define the per-token ratio as $\rho_t = \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}$, making the sequence-level ratio $\rho_{\text{seq}} = \prod_t \rho_t$.

In practice, we want to bound these ratios to maintain a trust region—preventing extreme weights that destabilize training. Two common approaches:

**Truncated Importance Sampling (TIS)** clips ratios to prevent extreme weights:
$$
\rho \leftarrow \text{clip}(\rho, C_{\min}, C_{\max})
$$

**Masked Importance Sampling (MIS)** discards samples with extreme ratios entirely:
$$
\rho \leftarrow \begin{cases} \rho & \text{if } C_{\min} \leq \rho \leq C_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

Truncation provides a soft constraint—we retain the gradient signal but limit its magnitude. Masking enforces a hard trust region where out-of-bounds samples are completely discarded.

Both approaches can be applied at token-level (per-token ratios) or sequence-level (single ratio per sequence). The choice matters more than you might think.

## the length bias problem

Here's where things get interesting. The sequence-level importance ratio $\rho_{\text{seq}} = \prod_t \rho_t$ grows or shrinks exponentially with sequence length.

Consider a modest per-token drift of 0.1%—that is, $\rho_t \approx 1.001$ on average. Over 2000 tokens:

$$
\rho_{\text{seq}} = (1.001)^{2000} \approx 7.39
$$

Even with minimal per-token divergence, long sequences get systematically rejected or heavily down-weighted. This creates a bias against longer responses regardless of their quality. The effective trust region varies dramatically with sequence length, and the model may learn to generate shorter sequences to avoid penalties—a form of reward hacking.

## geometric sequence masking

Geometric Sequence Masking addresses this by using the geometric mean of importance ratios instead of their product. The geometric mean captures an average per-token divergence that is independent of sequence length:

$$
\rho_{\text{geo}} = \left( \prod_{t=0}^{T-1} \rho_t \right)^{1/T}
$$

In log-space, this is simply the arithmetic mean of log-ratios:

$$
\log \rho_{\text{geo}} = \frac{1}{T} \sum_{t=0}^{T-1} \log \rho_t = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t|x, y_{<t})}{\mu(y_t|x, y_{<t})}
$$

This quantity represents the average per-token log-likelihood ratio, directly related to per-token KL divergence. The key property is length invariance: the geometric mean doesn't scale with $T$, making acceptance criteria consistent regardless of generation length.

The geometric sequence mask applies a two-sided threshold:

$$
g_{\text{geo-mask}} = \mathbf{1}\left[ C_{\min} \leq \rho_{\text{geo}} \leq C_{\max} \right]
$$

In code, implementing geometric masking requires only normalizing by sequence length. Starting from a standard importance sampling calculation:

```python
per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
per_seq_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)

# Add this line for geometric mean
per_seq_logps_diff = per_seq_logps_diff / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

importance_sampling_ratio = torch.exp(per_seq_logps_diff)
```

That single division by sequence length transforms length-biased sequence IS into length-invariant geometric masking.

## deepseek opsm

DeepSeek-V3.2 introduces Off-Policy Sequence Masking to stabilize training by masking sequences that have drifted too far from the current policy. The OPSM masking rule introduces a binary mask $M_{i,t}$:

$$
M_{i,t} = \begin{cases}
0 & \text{if } \hat{A}_{i,t} < 0 \quad \text{and} \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_{\text{old}}(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})} > \delta \\
1 & \text{otherwise}
\end{cases}
$$

The masking condition keeps a sequence if either: (1) it has non-negative advantage—we want to reinforce it regardless of drift, or (2) its KL divergence is within the threshold—it's still on-policy enough.

The rationale from the paper:

> Models benefit the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process.

This is a one-sided condition: it only masks negative-advantage samples that have drifted too far.

There's a crucial detail in the DeepSeek paper regarding notation:

> Note that $\pi_{\text{old}}$ here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above.

DeepSeek uses the inference engine's log-probs as $\pi_{\text{old}}$, meaning OPSM addresses both training-inference mismatch and policy staleness simultaneously.

## opsm is geometric sequence masking

Now for the punchline. The OPSM KL term is exactly the negative of the log geometric mean of importance ratios:

$$
\underbrace{\frac{1}{T} \sum_{t=0}^T \log \frac{\mu_{\text{old}}(y_t | x, y_{<t})}{\pi(y_t | x, y_{<t})}}_{\text{OPSM KL term}} = -\underbrace{\frac{1}{T} \sum_{t=0}^T \log \frac{\pi(y_t | x, y_{<t})}{\mu_{\text{old}}(y_t | x, y_{<t})}}_{\log \rho_{\text{geo}}}
$$

More simply: $\text{OPSM KL term} = -\log \rho_{\text{geo}}$.

We can rewrite OPSM's condition in terms of the geometric mean. The condition

$$
\frac{1}{T} \sum_{t=0}^T \log \frac{\mu_{\text{old}}}{\pi} > \delta
$$

is equivalent to:

$$
-\log \rho_{\text{geo}} > \delta \quad \Leftrightarrow \quad \log \rho_{\text{geo}} < -\delta \quad \Leftrightarrow \quad \rho_{\text{geo}} < e^{-\delta}
$$

OPSM is a one-sided geometric sequence mask that drops sequences where the current policy assigns significantly lower probability than the sampling distribution—the policy has moved away from these samples.

| Aspect | Geometric Sequence Mask | DeepSeek OPSM |
|--------|------------------------|---------------|
| Direction | Bidirectional ($C_{\min} \leq \rho_{\text{geo}} \leq C_{\max}$) | One-sided ($\rho_{\text{geo}} \geq e^{-\delta}$) |
| Advantage condition | None | Only applies to negative advantages |
| Distribution ratio | $\frac{\pi_{\text{old}}}{\mu_{\text{old}}}$ | $\frac{\pi}{\mu_{\text{old}}}$ |

The key difference in the distribution ratio is that OPSM conflates both sources of off-policyness—training-inference mismatch and policy staleness—into a single masking condition. We can make this explicit by factoring the OPSM ratio:

$$
\frac{\pi(y_t | x, y_{<t})}{\mu_{\text{old}}(y_t | x, y_{<t})} = \underbrace{\frac{\pi_{\text{old}}(y_t | x, y_{<t})}{\mu_{\text{old}}(y_t | x, y_{<t})}}_{\text{training-inference mismatch}} \times \underbrace{\frac{\pi(y_t | x, y_{<t})}{\pi_{\text{old}}(y_t | x, y_{<t})}}_{\text{policy staleness}}
$$

Taking logs and averaging over the sequence:

$$
\log \rho_{\text{geo}}^{\text{OPSM}} = \underbrace{\frac{1}{T} \sum_{t=0}^T \log \frac{\pi_{\text{old}}}{\mu_{\text{old}}}}_{\text{training-inference term}} + \underbrace{\frac{1}{T} \sum_{t=0}^T \log \frac{\pi}{\pi_{\text{old}}}}_{\text{staleness term}}
$$

This factorization is useful because the training-inference term is constant per rollout—it only needs to be computed once when data is generated. The staleness term changes with each gradient step as $\pi$ updates. In implementation:

```python
# Computed once per rollout
per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
per_seq_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
sampling_mean_kl = per_seq_logps_diff / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

# Computed per gradient step
staleness_kl = (per_token_logps - old_per_token_logps) * mask
staleness_mean_kl = staleness_kl.sum(dim=1) / mask.sum(dim=1)

# Total off-policy measure
total_mean_kl = staleness_mean_kl + sampling_mean_kl

# OPSM mask: keep if positive advantage OR low KL
is_pos_adv = advantages >= 0
is_low_kl = -total_mean_kl <= delta
opsm_mask = is_pos_adv | is_low_kl
```

## summary

The mathematical relationships are:

1. **Sequence IS** uses $\prod_t \rho_t$—length-biased, exponential growth
2. **Geometric masking** uses $(\prod_t \rho_t)^{1/T}$—length-invariant
3. **OPSM KL term** equals $-\log \rho_{\text{geo}}$—negative of log geometric mean
4. **OPSM** = geometric mask applied one-sided to $\frac{\pi}{\mu_{\text{old}}}$, conditioned on negative advantages
5. **Factored OPSM**: $\frac{\pi}{\mu_{\text{old}}} = \frac{\pi_{\text{old}}}{\mu_{\text{old}}} \times \frac{\pi}{\pi_{\text{old}}}$

When should you use what? Token-level IS when you want fine-grained control and can tolerate higher variance. Sequence-level geometric masking when you want length-invariant training-inference correction. OPSM when you want to handle both training-inference mismatch and policy staleness simultaneously, especially with multiple gradient steps per rollout.

The insight that OPSM is a geometric sequence mask isn't just mathematical trivia. It connects DeepSeek's technique to a principled framework for length-invariant importance sampling, clarifies what the method is actually optimizing, and provides a cleaner foundation for implementing variants and extensions.
