
---
## Qwen 3 235B A22B

This model was first released back in April, but got an update two days ago on July 22th, and has reclaimed the crown spot on benchmarks, reclaiming the spot from Kimi K2. The Qwen 3 family was released with 6 dense models and 2 MoE models. Their largest being this one which I'm discussing here. The model details:

- Model size is 235B (4.25x smaller than K2)
- 22B active parameters (1.5x fewer than K2)
- Vocabulary size 151k
- 8 active experts (no shared)
- Qwen2.5-MoE
- Grouped Query Attention
- 36 Trillion Tokens!!

![Image](https://pbs.twimg.com/media/GwaIVJRX0AASdq3?format=jpg&name=medium)

I want to go through the architecture and check out any interesting deviations from K2. Obviously the size difference between the models is stark. In a broad sense, the architectures are fairly similar, MoE architectures seem to have converged slightly. Obviously Qwen3 is much smaller than K2, so the hidden dim is considerably smaller, the MLP dimension aswell, Qwen only has 128 experts as opposed to 384. They both have 64 attention heads, although obviously K2 use MLA while Qwen3 use GQA. Both use exactly the same SwiGLU as their FFNs. Another big deviation is the fact that Qwen 3 alternates between Dense and MoE. 

## Kimi K2
Moonshot released Kimi K2 last week, its the first open weight 1T model and has been crushing benchmarks. It's not a reasoning model per se but has been extensively trained with RL. The model architecture builds **heavily** on DeepSeek V3 

* Model size is 1T 
* 32B active parameters
* DeepSeekMoE (from DeepSeekV3)
* Vocabulary size 160k
* 1 shared + 8 active experts
* 15.5 Trillion tokens


![[Pasted image 20250715140635.png]]

Kimi has yet to release their technical report, but there are a few interesting forum posts on Zhihu (Chinese Quora) from team members. 

### Shaowei Liu
is a low-key infra engineer at MoonShot sharing a inference perspective on K2 design choices.

Moonshot ran many scaling law experiments on architecture variants, and every single variant that differed from DSv3 failed to beat it - at best they tied. So, should they differ to an inferior architecture just for the sake of being different? **No.** DSv3 is battle tested at scale and Moonshots own ideas weren't. They were already set on using Muon optimizer and a larger parameter count, so they didn't want the risk of a third unverified variable. So they boiled down to inheriting DSv3 architecture, and find parameters that keep train/inference cost flat while pushing loss lower than DS did.

#### Deviations from DSv3
As you can observe in the picture above, there are four deviations in the arch config

**Experts 256 -> 584**
With a fixed activated params, increasing the total MoE params still obeys the scaling laws meaning this is how K2 pushes the loss lower.

During pre-fill they are still compute-bound. Activated params & FLOPs are unchanged -> no extra time.

During decoding the GEMM becomes memory-bound at production batch sizes which means that an increase of 1.5x in params leads to 1.5x memory traffic.

**Attention heads 128 -> 64**
MoE just got 50% more expensive - can we claw it back elsewhere?

DeepSeek doubles the num heads vs classic MHA to max bandwidth utilization, K2 rolls this back to 64. 

Prefill: Attention FLOPs ∝ `h s²`. Cutting heads **halves the quadratic term**—huge win for long sequences (our bread & butter: agents, vibe coding).
Decode: KV-cache size is unchanged but the QKVO activation traffic drops 5GB.

Ablations show negligible impact on loss compared to MoE's gain. 

**Only the first layer is dense (first_k_dense=3->1)**

Just like DeepSeek, K2 observe that load imbalance is very big for the first layer, **but only for the first layer**. DeepSeek had 3 dense layers to begin with while K2 only starts with a single dense.

**Router has no expert grouping (n_group=8->1)**
Grouping helps when multiple experts sit on one GPU, balancing work at device level. However, at K2's scale they use a large EP so each device holds =< 1 expert. Balancing moves to the node level and as such this parameter is no longer useful.

---

Those four tiny knobs yield an inference recipe that:

- Keeps EP count identical to DSv3.
- Grows total params **1.5×** yet, **ignoring comms**, shows **lower theoretical prefill & decode time**.
- Even after real-world comm overlap, cost stays **no worse** than DSv3.

Each change was backed by **solid theory + ablation**. Once K2 is fully open-sourced, we hope the wider inference community will stress-test these claims.

### Tech report 
is now released, reading it now and reporting interesting findings.


**Moun**
Obviously, the Moun variant MounClip optimizer that is used is the most interesting part of K2. They manage to train for 15 trillion tokens without loss spikes. I'm not well versed in the optimizer space though so I won't comment more on this part of K2.

**Pre-training data**
K2 uses a synthetic data generation pipeline that carefully rephrases data to amplify the volume of high quality tokens without introducing overfitting. Pre-training on natural, knowledge-intensive text presents a trade-off: a single epoch is insufficient for comprehensive knowledge absorption, while multi-epoch repetition yields diminishing returns and increases the risk of overfitting. Instead, they build a pipeline that rephrases the data, and avoid training over multiple epochs on the same data.

**Sparisty Scaling Law**
Sparsity is defined as the ratio between the total number of experts and the number of active experts. A clear goal for K2 was not to increase the compute resources, as observed from above discussions. But, in order to squeeze more performance out of the resources they had, they performed controlled experiments of different sparsity factors. They observe that - under a fixed number of activated parameters (i.e constant FLOPS) - increasing the total number of experts (i.e increasing sparisty) consistently lowers both training and validation loss. This is clearly observed below.
![[Screenshot 2025-07-24 at 16.04.10.png]]
However, while increased sparsity leads to better performance, it comes with increase infrastructure complexity. To balance these metrics, K2 adopts a sparsity of 48, activating 8 out of 384 experts per forward pass.

**Number of attention heads**
DeepSeek V3 sets the number of attention heads to roughly twice the number of model layers to better utilize memory bandwidth and enhance computational efficiency. However, this comes at a serious inference premium at longer context lengths. For example, at 128k ctx len, doubling the attention heads from 64 to 128 (DSv3) is an 83% increase in inference FLOPs. Again, Kimi team perform controlled experiments, investigating the performance improvements from the increase in attention heads to see if the tradeoff is worth it. 
![[Screenshot 2025-07-24 at 16.10.45.png]]
We see that the performance gain for this doubling is minor, and therefor note pursued by K2 - which uses 64 attention heads. 