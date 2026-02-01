# Research Notes: Geometric Sequence Masking and DeepSeek OPSM

This document compiles research notes for writing a blog post on Geometric Sequence Masking (GSM), its relationship to DeepSeek's Off-Policy Sequence Masking (OPSM), and how OPSM can be understood as a variant of GSM. The goal is to provide mathematical foundations and practical context for implementing these techniques in RL training for LLMs.

---

## Table of Contents

1. [Theory Background: The Off-Policy Problem in RL for LLMs](#1-background-the-off-policy-problem-in-rl-for-llms)
2. [Existing Importance Sampling Correction Methods](#2-existing-importance-sampling-correction-methods)
3. [The Length Bias Problem](#3-the-length-bias-problem)
4. [Geometric Sequence Masking (GSM)](#4-geometric-sequence-masking-gsm)
5. [DeepSeek OPSM](#5-deepseek-opsm)
6. [OPSM as a Variant of Geometric Sequence Masking](#6-opsm-as-a-variant-of-geometric-sequence-masking)
7. [Factoring OPSM: Training-Inference Mismatch + Staleness](#7-factoring-opsm-training-inference-mismatch--staleness)
8. [Key Takeaways and Comparison Table](#8-key-takeaways-and-comparison-table)
9. [References](#9-references)

---

## 1. Theory Background: The Off-Policy Problem in RL for LLMs

### PPO 
Proximal Policy Optimization (PPO) introduces a clipped surrogate objective:
$$
J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).
$$

Here, $\pi_\theta(a|s)$ is the current policy being optimized and $\pi_{\theta_{\text{old}}}(a|s)$ is the policy that was used to collect the training data (i.e., the policy from the previous iteration). The ratio between these two policies emerges from *importance sampling*, which allows us to reuse data collected under an old policy to estimate gradients for a new policy.

Recall from the advantage formulation of the policy gradient that we have:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right].
$$ 

This expectation is taken over trajectories sampled from $\pi_\theta$, but in practice we want to take multiple gradient steps on a batch of data that was collected from a fixed policy $\pi_{\theta_{\text{old}}}$. To correct for this distribution mismatch, we multiply by the importance weight $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$, which reweights samples to account for how much more or less likely they are under the current policy versus the data-collection policy.

This correction method which enables multiple gradients steps per batch is crucial for efficiency, and is persistent in all of the RL algorithms we see applied in the LLM post training chain today. Being able to generate a large rollout batch is key to achieving high levels of hardware utilization, minimizing acceleartor ideling. GRPO, which inherits PPO clipping naturally retains this functionality, but even methods like CISPO that move closer to a vanilla REINFORCE objective still retain this distribution correction for offline updates. 

### Batch-size invariant PPO

A few years later, Schulman and friends published a follow upp paper to PPO. The authors argue that standard policy optimization algorithms (like PPO) conflate two distinct roles into a single "old" policy, $\pi_{\text{old}}$. They propose decoupling these roles into two separate policies: the **behavior policy** $\mu$ and the **proximal policy** ($\pi_{\text{prox}}$). This turns the objective into a three policy formulation.

#### 1. Behavior Policy ($\mu$)
*   **Role:** **Off-policy correction**.
*   **Definition:** This is the specific policy instance that was used to interact with the environment and collect the experience data (the "rollout").
*   **Why it matters:** Because the current policy $\pi_{\theta}$ changes during optimization, the data becomes "stale" or "off-policy." To compute unbiased gradient estimates, we must correct for the mismatch between the data-collecting distribution and the current optimization distribution.
*   **Mechanism:** It appears in the denominator of the **importance sampling ratio**:
    $$ \frac{\pi_{\theta}(a|s)}{\mu(a|s)} $$
*   This is the importance sampling weight that corrects for behaviour policy. The behaviour policy is frozen during training.
    

#### 2. Proximal Policy ($\pi_{\text{prox}}$)
*   **Role:** **Update control (Trust Region)**.
*   **Definition:** This is a recent version of the policy used as an anchor to prevent the new policy from changing too quickly or drastically during an update.
*   **Why it matters:** Policy optimization is unstable if the policy shifts too far in a single step (the "trust region" concept). We need to constrain the update size.
*   **Mechanism:** In the authors' proposed **decoupled objective**, this policy is used in the clipping term (or KL penalty) to limit how far $\pi_{\theta}$ deviates:
    $$ \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\text{prox}}(a|s)}, 1-\epsilon, 1+\epsilon\right) $$
    Crucially, the authors show this policy **does not** need to be the same as the behavior policy. It just needs to be a "recent enough" policy to ensure stable learning dynamics.

The decoupled PPO objective 

$$
J(\theta)_{\text{decoupled}} =  \frac{\pi_{{\text{prox}}}(a | s)}{\mu(a | s)} \min\left(\frac{\pi_\theta(a|s)}{\pi_{{\text{prox}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{{\text{prox}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right)
$$

In standard PPO, these two are forced to be identical ($\pi_{\text{old}} = \mu = \pi_{\text{prox}}$). This makes the algorithm sensative to batch size. Decoupling the policies means we free us from this constraint, because the update control is independent of the data aggregation. Additionally, it introduces a more flexible behaviour policy behaviour. We can now introduce multiple workers, replay buffers, stable checkpoints as our behaviour policy while retaining the control size update inside the trust region.

#### Group Relative Policy Optimization
Is a PPO-inspired algorithm with a simliar clipped surrogate objective, but that does away with the learning of the value function and instead performs value estimation by estimating the advantage or baseline by collecting multiple completions and rewards from the same initial prompt, i.e it performs Monte Carlo estimates. 

The GRPO objective (leaving out KL divergence for simplicity):

$$
J(\theta) = \frac{1}{G}\sum_{i=1}^G \min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right).
$$

with the advantage computation

$$
A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.
$$

[needs more]

### Dissaggregated RL

Modern RL training for LLMs use a hybrid setup where:
- **Inference/Sampling**: A high-throughput inference engine (e.g., vLLM) generates rollouts
- **Training**: A separate training framework (PyTorch, DeepSpeed, etc.) performs gradient updates

This separation is driven by efficiency—inference engines like vLLM are highly optimized for throughput during generation, while training frameworks need different optimizations for gradient computation.

This hybrid setup introduces **two distinct sources of off-policyness**:

1. **Training-Inference Mismatch**: The sampling distribution from the inference engine ($\mu$) may differ from the training model's distribution ($\pi_{\text{old}}$) due to:
   - Quantization differences
   - Different attention implementations
   - Numerical precision differences
   - Different softmax implementations

2. **Policy Staleness**: Between when a sequence is sampled and when it's used for training, the policy may have changed due to gradient updates. With multiple gradient steps per generation batch, later mini-batches train on data that's increasingly "stale" relative to the current policy.

> "Additionally, inference frameworks used for efficient data generation are often highly optimized, which may differ in implementation details from training frameworks. Such training-inference inconsistency further exacerbates the degree of off-policyness."
> — DeepSeek-V3.2 Technical Report

AReaL (https://arxiv.org/abs/2505.24298) proposed the Decoupled PPO formulation but this was primarily to support the asynchronous setup with training batches being produced by multiple policy versions leading to a distribution gap between the data and the current policy $\pi$. This breaks the classical PPO assumption that all data is generated by $\pi_\text{old}$. 

However, even in the synchronous setup where the data is generated synchronously at a single point in time, by what in theory is $\pi_\text{old}$, it turns out that the dissagregated nature of the setup actually turns this into an off-policy setting anyway, with $\pi_\text{old} \neq \mu$. MiniMax actually noted this in their M1 paper (https://arxiv.org/abs/2506.13585), but it wasn't until this blog (https://fengyao.notion.site/off-policy-rl) in August last year until someone properly studied the problem. Their proposal was a truncated importance weight, more on this later. NeMo RL also proposed the same importance sampling weight in their repo back in April 2025 (https://github.com/NVIDIA-NeMo/RL/pull/174).

### Notation Convention

Throughout this document, we use the following notation, following a three policy formulation that is employed in both `verl` and `TRL`:

| Symbol | TRL Variable | Description |
|--------|--------------|-------------|
| $\pi$ | `per_token_logps` | Log-probs of the **current policy** being optimized |
| $\pi_{\text{old}}$ (proximal polixy) | `old_per_token_logps` | Log-probs from the **training model at sampling time** (start of training step) |
| $\mu_{\text{old}}$ (behaviour policy)| `sampling_per_token_logps` | Log-probs from the **inference engine** at sampling time |

The **training-inference mismatch** is captured by the ratio $\frac{\pi_{\text{old}}}{\mu_{\text{old}}}$.

---

## 2. Existing Importance Sampling Correction Methods

### The Standard IS Framework

The standard way to correct for distribution shift is **importance sampling (IS)**. Given samples from a behavior policy $\mu$, we can estimate expectations under a target policy $\pi$ using importance weights:

$$
\mathbb{E}_{y \sim \pi}[f(y)] = \mathbb{E}_{y \sim \mu}\left[\frac{\pi(y)}{\mu(y)} f(y)\right]
$$

For autoregressive generation, the sequence-level importance ratio is:

$$
\frac{\pi(y|x)}{\mu(y|x)} = \prod_{t=0}^{T-1} \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

For the three policy formulation, we directly target the distribution shift between the proximal policy $\pi_\text{old}$ and the behaviour policy $\mu_\text{old}$

$$
\frac{\pi_\text{old}(y|x)}{\mu_\text{old}(y|x)} = \prod_{t=0}^{T-1} \frac{\pi_\text{old}(y_t | x, y_{<t})}{\mu_\text{old}(y_t | x, y_{<t})}
$$

### Per-token vs Sequence-level

We can define the importance ratio at a token-level, as such:

$$
\rho_t = \frac{\pi_{\text{old}}(y_t| x, y_{< t })}{\mu_{\text{old}} (y_t | x, y_{< t})}
$$

or at the sequence-level as a product of per-token ratios:

$$
\rho_{\text{seq}} = \prod_{t=0}^{T-1} \rho_t
$$

[introducing log formulations of IS, for autoregressive formulations, token level sequence level may perhaps help understand connections to OPSM later]

### Practice
In practice, we've seen variants of these importance ratio that attempt to enforce trust regions through either clipping or masking. This is motivated by the idea that there are OOD samples that we want to avoid, its a training stability thing.

**Truncated Importance Sampling (TIS)**: Clips ratios to prevent extreme weights

$$
\rho \leftarrow \text{clip}(\rho, C_{\min}, C_{\max})
$$

**Masked Importance Sampling (MIS)**: Discards samples with extreme ratios

$$
\rho \leftarrow \begin{cases} \rho & \text{if } C_{\min} \leq \rho \leq C_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

 These provide a soft vs hard way to enfore the trust region. In truncation, we down-weight the samples outside the trust region, we retain the signal but its clipped, this is greater sample efficiency but if there are true OOD samples, we'll still retain their gradients. Masking enforces a hard trust region where samples outside of the region are completely discarded.

 The choice of trust region enforcement is orthogonal to the choice of leve formulation, both can be applied at **token-level** (per-token ratios) or **sequence-level** (single ratio per sequence). 

The IS correction methods are all applied as a importance sampling weight between the proximal policy and the behaviour policy: $\frac{\pi_{\text{old}}}{\mu_{\text{old}}}$, directly targeting the training-inference mismatch, following the three policy formulation. 

Remember, IS correct is applicable to any objective, it just happens that we often discuss it in relation to GRPO. Here's an example of truncated importance sampling applied to the GRPO objective.

$$
J(\theta) = \frac{1}{G}\sum_{i=1}^G \text{clip}\bigl(\frac{{\pi_{\text{old}}}(a_i | s)}{{\mu_\text{old}}(a_i | s)}, C_{\min}, C_{\max}\bigr) \min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right).
$$

### Code

Calculating either masked or truncated importance sampling weights at token or sequence level is straightforward. Assuming here that 

$\pi_\text{old}$ = `old_per_token_logps`

$\mu_\text{old}$ = `sampling_per_token_logps`

```
per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

if sequence_level_is:
    per_seq_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
    importance_sampling_ratio = torch.exp(per_seq_logps_diff)
else:
    importance_sampling_ratio = torch.exp(per_token_logps_diff)

# importance_sampling_ratio.shape:
#  (B, T)  for per-token variants
#  (B, 1)  for per-sequence variants

if truncate:
    importance_sampling_weight = torch.clamp(importance_sampling_ratio, min=C_min, max=C_max,)
elif mask:
    invalid_mis_mask = (importance_sampling_ratio < C_min) | (importance_sampling_ratio > C_max)
    importance_sampling_weight = importance_sampling_ratio.masked_fill(invalid_mis_mask, value=0.0)
```

## 3. The Length Bias Problem

### Exponential Growth with Sequence Length

The fundamental problem with sequence-level importance ratios is that they grow (or shrink) **exponentially** with sequence length.

Consider a modest per-token drift of 0.1% (i.e., $\rho_t \approx 1.001$ on average). Over 2000 tokens:

$$
\rho_{\text{seq}} = (1.001)^{2000} \approx 7.39
$$

This means even with minimal per-token divergence, long sequences get systematically rejected or heavily down-weighted.

### The Problem for RL Training

This length bias creates several issues:

1. **Systematic bias against longer sequences**: Longer responses are more likely to be masked or clipped, regardless of their quality
2. **Inconsistent trust regions**: The effective constraint varies dramatically with sequence length
3. **Reward hacking**: The model may learn to generate shorter sequences to avoid penalties

> "Standard importance sampling for autoregressive generation computes $\rho(y) = \prod_{t=0}^{T-1} \rho_t$. This product grows exponentially even with modest per-token divergence."
> — Richard Li, "RL Collapse Part 3"

---

## 4. Geometric Sequence Masking (GSM)

### The Core Insight

Geometric Sequence Masking addresses the length bias problem by using the **geometric mean** of importance ratios instead of their product.

The geometric mean captures an **average per-token divergence** that is independent of sequence length:

$$
\rho_{\text{geo}} = \left( \prod_{t=0}^{T-1} \rho_t \right)^{1/T}
$$

### Log-Space Formulation

In log-space, this is simply the arithmetic mean of log-ratios. Applied to address our training-inference distribution shift we get:

$$
\log \rho_{\text{geo}} = \frac{1}{T} \sum_{t=0}^{T-1} \log \rho_t = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi_\text{old}(y_t|x, y_{<t})}{\mu_\text{old}(y_t|x, y_{<t})}
$$

This quantity represents the **average per-token log-likelihood ratio**, which is directly related to the per-token KL divergence.

### Key Properties of GSM

1. **Length Invariance**: The geometric mean doesn't scale with $T$, making acceptance criteria independent of generation length

2. **Per-Token Trust Region**: The constraint bounds average per-token KL divergence rather than cumulative divergence


### GSM Masking Rule

The geometric sequence mask applies a two-sided threshold:

$$
g_{\text{geo-mask}} = \mathbf{1}\left[ C_{\min} \leq \rho_{\text{geo}} \leq C_{\max} \right]
$$

Or equivalently in log-space:

$$
g_{\text{geo-mask}} = \mathbf{1}\left[ \log C_{\min} \leq \log \rho_{\text{geo}} \leq \log C_{\max} \right]
$$

### Code

Extending the previous implementation to implement Geo-Mask requires us only to introduce two new lines of code where we normalize by the sequence length $\frac{1}{T}$. GSM would take the mask branch in the later conditional branch.
```
per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

if sequence_level_is:
    per_seq_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
--> if geometric_mean:
-->     per_seq_logps_diff = per_seq_logps_diff / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    importance_sampling_ratio = torch.exp(per_seq_logps_diff)
else:
    importance_sampling_ratio = torch.exp(per_token_logps_diff)

# importance_sampling_ratio.shape:
#  (B, T)  for per-token variants
#  (B, 1)  for per-sequence variants

if truncate:
    importance_sampling_weight = torch.clamp(importance_sampling_ratio, min=C_min, max=C_max,)

elif mask:
    invalid_mis_mask = (importance_sampling_ratio < C_min) | (importance_sampling_ratio > C_max)
    importance_sampling_weight = importance_sampling_ratio.masked_fill(invalid_mis_mask, value=0.0)
```


---

## 5. DeepSeek OPSM

### Overview

DeepSeek-V3.2 introduces **Off-Policy Sequence Masking (OPSM)** to stabilize training by masking sequences that have drifted too far from the current policy.

From the paper:
> "To improve the efficiency of RL systems, they typically generate a large batch of rollout data, which is subsequently split into multiple mini-batches for several gradient update steps. This practice inherently introduces off-policy behavior."

### The OPSM Masking Rule (Equation 9)

OPSM introduces a binary mask $M_{i,t}$ into the GRPO loss:

$$
M_{i,t} = \begin{cases}
0 & \text{if } \hat{A}_{i,t} < 0 \quad \text{and} \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_{\text{old}}(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})} > \delta \\
1 & \text{otherwise}
\end{cases}
$$

where:
- $\hat{A}_{i,t}$ is the advantage
- $\delta$ is a threshold hyperparameter
- $|o_i|$ is the sequence length

The masking condition keeps a sequence if either:
1. It has non-negative advantage ($\hat{A} \geq 0$) — we want to reinforce it regardless of drift
2. Its KL divergence is within the threshold — it's still on-policy enough

The rationale:
> "Models benefit the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process."

This is a **one-sided** condition.

Crucially, from the DeepSeek paper:

> "Note that $\pi_{\text{old}}$ here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above."

In other words, DeepSeek uses the **inference engine's log-probs** ($\mu_{\text{old}}$ in our notation) as $\pi_{\text{old}}$, which means OPSM addresses **both** training-inference mismatch AND policy staleness simultaneously. 

The OPSM KL formulation looks similar to what we've been formulating previously. Using our notation

$$
\frac{1}{T}\sum_{t=0}^{T-1} \log \frac{\mu_\text{old}(y_t|x, y_{<t})}{\pi(y_t|x, y_{<t})}
$$

this looks very similar to the Geo-mask, but with a different ratio, now incorporatting the current policy $\pi$.

Previous GRPO formulations have let the ratio within the PPO-clip address the policy staleness distribution shift. The OPSM formulation retains this behaviour, but takes an even safer approach when it incorporates the current policy as part of this binary mask as well. Essentially it covers for any kind of off-policyness that is introduced, saying that we want to hard enforce the trust region.

---

## 6. OPSM as a Variant of Geometric Sequence Masking

### The Mathematical Equivalence

The key observation connecting OPSM to GSM is that OPSM's KL divergence term is **exactly the negative of the log geometric mean** of importance ratios:

$$
\underbrace{\frac{1}{T} \sum_{t=0}^T \log \left( \frac{\mu_{\text{old}} (y_t | x, y_{ < t})}{\pi(y_t| x, y_{ < t })} \right)}_{\text{DeepSeek OPSM}} = - \underbrace{\frac{1}{T} \sum_{t=0}^T \log \left( \frac{\pi(y_t| x, y_{ < t })}{\mu_{\text{old}} (y_t | x, y_{ < t})} \right)}_{\text{Geo Mask}}
$$

Or more simply:

$$
\text{OPSM KL term} = -\log \rho_{\text{geo}}
$$

but importantly this still "conflates" both training-inference, and policy staleness into a single masking condition, unlike our previous geo-mask which was only targeted at the training-inference shift.

### Rewriting OPSM's Condition

OPSM's masking condition can thus be rewritten in terms of the geometric mean:

$$
\frac{1}{T} \sum_{t=0}^T \log \frac{\mu_{\text{old}}}{\pi} > \delta
$$

is equivalent to:

$$
-\log \rho_{\text{geo}} > \delta \quad \Leftrightarrow \quad \log \rho_{\text{geo}} < -\delta \quad \Leftrightarrow \quad \rho_{\text{geo}} < e^{-\delta}
$$

So OPSM is a **one-sided geometric sequence mask** that drops sequences where the current policy assigns significantly lower probability than the sampling distribution (i.e., the policy has moved away from these samples).

### Comparison: GSM vs OPSM

GSM applies beyond just the training inference mismatch, but if we compare the GSM implementation from previous to OPSM it looks like this:
| Aspect | Geometric Sequence Mask (GSM) | DeepSeek OPSM |
|--------|------------------------------|---------------|
| Direction | Bidirectional ($C_{\min} \leq \rho_{\text{geo}} \leq C_{\max}$) | One-sided ($\rho_{\text{geo}} \geq e^{-\delta}$) |
| Advantage condition | None | Only applies to negative advantages |
| What it measures | Training-inference mismatch | Training-inference mismatch + policy staleness |
| Distribution ratio | $\frac{\pi_{\text{old}}}{\mu_{\text{old}}}$ | $\frac{\pi}{\mu_{\text{old}}}$ |

---

## 7. Factoring OPSM: Training-Inference Mismatch + Staleness

Something that is interesting, and eye opening, is how we express DS OPSM as a form of geometric sequence masking. We understand Geo-masking and how to apply it to the training-inference mismatch, but we can also express OPSM as two geometric mean terms, and bridge our understanding between the proposed OPSM KL, and Geo-masking. 

Again, remember that we can reformulate OPSM as geo-mask between $\pi$ and $\mu$:

$$
\underbrace{\frac{1}{T} \sum_{t=0}^T \log \left( \frac{\mu_{\text{old}} (y_t | x, y_{ < t})}{\pi(y_t| x, y_{ < t })} \right)}_{\text{DeepSeek OPSM}} = - \underbrace{\frac{1}{T} \sum_{t=0}^T \log \left( \frac{\pi(y_t| x, y_{ < t })}{\mu_{\text{old}} (y_t | x, y_{ < t})} \right)}_{\text{Geo Mask}}
$$

Then, to get there we start by realizing that OPSM's importance ratio can be factored into two components (borrowing from the formulation presented in https://arxiv.org/abs/2512.01374v1):

$$
\frac{\pi(y_t| x, y_{< t })}{\mu_{\text{old}} (y_t | x, y_{ < t})} = \underbrace{\frac{\pi_{\text{old}}(y_t| x, y_{< t })}{\mu_{\text{old}} (y_t | x, y_{ < t})}}_{\text{training-inference mismatch } (\rho_t)} \times \underbrace{\frac{\pi(y_t| x, y_{< t })}{\pi_{\text{old}}(y_t| x, y_{ < t })}}_{\text{policy staleness}}
$$

This factorization is significant because:

1. Our previous formulation only addresses training-inference mismatch through $\frac{\pi_{\text{old}}}{\mu_{\text{old}}}$

2. **DeepSeek's OPSM** conflates both sources, solving for both simultaneously

3. **The factored form** makes it explicit that OPSM is handling two distinct phenomena

### In Log-Space

Taking logs and averaging over the sequence:

$$
-\log \rho_{\text{geo}}^{\text{OPSM}} = -\frac{1}{T} \sum_{t=0}^T \log \frac{\pi}{\mu_{\text{old}}}
$$

$$
= -\left( \underbrace{\frac{1}{T} \sum_{t=0}^T \log \frac{\pi_{\text{old}}}{\mu_{\text{old}}}}_{\log \rho_{\text{geo}}^{\text{TI}}} + \underbrace{\frac{1}{T} \sum_{t=0}^T \log \frac{\pi}{\pi_{\text{old}}}}_{\log \rho_{\text{geo}}^{\text{Stale}}} \right)
$$

where $\log \rho_{\text{geo}}^{\text{TI}}$ is the training-inference geometric mean that we know from before.

[this $\log \rho_{\text{geo}}^{\text{TI}}$ is a good notation but it isn't as clear that this term is exactly the geo-mask from before.]

### Rewriting OPSM Using This Factorization

This formulation:

1. Separates the two sources of off-policyness conceptually
2. During training, $\log \rho_{\text{geo}}^{\text{TI}}$ is frozen, it is calculated once per rollout, where only the staleness term needs to be computed per gradient step

### Code

Based on our reformulation, we retain the $\log \rho_{\text{geo}}^{\text{TI}}$ calculation from the Geo-mask code snipept from before. This looks exactly the same and is calculated once per rollout.

```
per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

if sequence_level_is:
    per_seq_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
    if geometric_mean:
        per_seq_logps_diff = per_seq_logps_diff / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

sampling_mean_kl_div = per_seq_logps_diff
```

Then, on each gradient step, we need to calculate $\rho_{\text{geo}}^{\text{Stale}}$

```python
# Computed per gradient step in _compute_loss:
staleness_kl = per_token_logps - old_per_token_logps  # policy staleness
staleness_mean_kl = (staleness_kl * mask).sum(dim=1) / mask.sum(dim=1)

# Total off-policy KL
total_mean_kl = staleness_mean_kl + sampling_mean_kl_div

is_pos_adv = advantages >= 0
is_low_kl = -total_mean_kl <= off_policy_threshold
off_policy_sequence_mask = (is_pos_adv | is_low_kl)
```


## 8. Key Takeaways and Comparison Table

### Mathematical Relationships

1. **Sequence IS** uses $\prod_t \rho_t$ — length-biased, exponential growth
2. **Geo-Mask** uses $(\prod_t \rho_t)^{1/T} = \exp(\frac{1}{T}\sum_t \log \rho_t)$ — length-invariant
3. **OPSM KL term** equals $-\log \rho_{\text{geo}}$ — negative of log geo-mean
4. **OPSM** = Geo-Mask applied one-sided to $\frac{\pi}{\mu_{\text{old}}}$, conditioned on negative advantages
5. **Factored OPSM**: $\frac{\pi}{\mu_{\text{old}}} = \frac{\pi_{\text{old}}}{\mu_{\text{old}}} \times \frac{\pi}{\pi_{\text{old}}}$

### When to Use What

- **Token-level IS**: When you want fine-grained control and can tolerate higher variance
- **Sequence Geo-Mask**: When you want length-invariant training-inference correction
- **OPSM**: When you want to handle both training-inference mismatch AND policy staleness, especially with multiple gradient steps per rollout
- **Factored OPSM**: When you want the benefits of OPSM but with clearer separation of concerns and ability to precompute the training-inference component

---

## 9. References

### Primary Sources

1. **DeepSeek-V3.2 Technical Report**
   - Paper: https://arxiv.org/abs/2512.02556
   - HTML: https://arxiv.org/html/2512.02556v1
   - Section 3.1: Off-Policy Learning, Equation 9

2. **Geometric Sequence Masking Blog Post**
   - Richard Li: https://richardli.xyz/post/rl-collapse-part3/
   - Covers the theory of length-invariant importance sampling

### TRL Implementation

4. **PR #4891: Geometric Sequence Masking**
   - https://github.com/huggingface/trl/pull/4891
   - Introduces GSM and refactors OPSM as a variant

5. **PR #4689: DeepSeek V3.2 Off-Policy Sequence Masking**
   - https://github.com/huggingface/trl/pull/4689
   - Original OPSM implementation

6. **TRL GRPO Documentation**
   - Training-inference mismatch: https://huggingface.co/docs/trl/grpo_trainer
   - Paper index (TIS/MIS): https://huggingface.co/docs/trl/paper_index

### Key Quotes from PR #4891

> "TRL's existing importance sampling correction (TIS/MIS) uses per-token ratios... When applied at the sequence-level (Sequence-level IS), this turns into a product of per-token ratios which is **length-biased**—systematically favoring shorter sequences in masking/truncation decisions."

> "DeepSeek's Off Policy Sequence Masking technique is a form of Geometric Sequence Masking to address both training-inference mismatch and policy staleness. It can be shown to be equivalent to negated Geometric Masking."

> "However, this expression conflates the training-inference mismatch, with the policy staleness, solving for both sources of off-policyness at the same time. This differs from TRL's existing IS correction (between $\pi_{\text{old}}$ and $\mu_{\text{old}}$) which only addresses training-inference mismatch."

### Key Quotes from PR #4689 Discussion

> "According to the paper, DeepSeek uses off-policy sequence masking to mitigate off-policy effects caused by both (1) policy staleness and (2) training–inference mismatch."

> "A key detail is that they set $\pi_{\text{old}}$ to the *inference-time sampling distribution*... This can be understood as sequence masking based on an importance ratio of the form $\frac{\pi_{\text{train}}(y \mid x)}{\pi_{\text{infer}}(y \mid x)}$."

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Training-Inference Mismatch** | Distribution shift between inference engine (vLLM) and training model due to implementation differences |
| **Policy Staleness** | Distribution shift caused by gradient updates between sampling and training |
| **Importance Sampling (IS)** | Technique to estimate expectations under one distribution using samples from another |
| **Geometric Mean** | The $n$-th root of the product of $n$ numbers; length-invariant for sequences |
| **TIS (Truncated IS)** | Clips importance ratios to bound values |
| **MIS (Masked IS)** | Discards samples with importance ratios outside bounds |
| **OPSM** | Off-Policy Sequence Masking from DeepSeek-V3.2 |
| **GSM** | Geometric Sequence Masking; uses geometric mean of ratios for length invariance |
| **KL Divergence** | Measure of difference between probability distributions |
| **Forward KL** | $D_{KL}(P \| Q) = \sum P \log(P/Q)$ — mode-seeking |
| **Trust Region** | Constraint on policy updates to ensure stability |

