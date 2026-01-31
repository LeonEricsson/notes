
has been a hot discussion over the past few months. All the way back to the MiniMax M1 paper we've know that using a separate inference framework (typically employed during RL) is problematic as these engines produce slightly different numerical outputs due to precision errors. This fundamentally changes the objective and risks training collapse. The first attempts to address this that we saw were through algorithmic patches based on importance sampling. We've seen truncated importance sampling and masked importance sampling as ways to correct the biased gradient introduced by the training-inference mismatch.

Several engineering based approaches have also been explored. MinMax proposed using a FP32 language model head. The Ling Team tried to manually align training and inference implementations, finding that this removed their mismatch, but this is a very costly approach that is difficult to generalize across the open space.

This new paper proposes a new fix. They investigate the root cause of the numerical mismatch: floating-point precision. BF16 is the default choice in neural training due to the wide dynamic range which is excellent for stable pre-training. It does however have a low precision which makes it susceptible to rounding errors. The authors find that simply switching BF16 to FP16 during RL fine-tuning virtually eliminates the training-inference mismatch.

#### FP16 vs BF16
When representing real numbers you divide your bit budget (16) between two components: exponent bits, which determin the range (how lare and small a value can be), and mantissa bits (also known as fraction bits), which determine the precision (how large the step is between two numbers). Both representations use 16 bits but they allocate them differently. BF16 uses 8 bits for the exponent, matching the range of the 32-bit FP32 format, and only 7 bits for the mantissa. This makes BF16 **highly** resistent to overflow and underflow, at the cost of reduced precision. This provides very high stability which has been key in large scale deep learning systems. FP16 uses 5 bits for the exponent and 10 bits for the mantissa. This means FP16 has a high numerical precision but at the cost of a severely constrained dynamic range. 

Findings are that switching over to FP16 completely basically removes the entire training-inference mismatch. Suggesting that perhaps RL fine-tuning should consider this switch.

*This is precisely why switching to FP16 provides a fundamental solution. With its 10 mantissa bits, FP16 offers 8 times more precision (2^10 values vs. 2^7 values) than BF16. This higher fidelity means that the outputs of the training and inference engines are much more likely to be numerically identical. The increased precision creates a buffer that absorbs the minor implementation differences between the two engines, preventing rounding errors from accumulating and causing a policy divergence.*

*For RL fine-tuning, the dynamic range of the model’s weights and activations has already been established during pre-training. Therefore, the extreme range of BF16 is less critical, while the precision it sacrifices becomes a dominant drawback. By reverting to FP16, we trade the unnecessary range of BF16 for the critical precision, effectively closing the gap between training and inference without any complex algorithmic or engineering workaround.*

![[Screenshot 2025-11-05 at 09.50.16.png|700]]

### Following discussion
Trying to capture the ensuing discussion after this paper was released. 

@shfx0072 performs similar offline comparisons that the paper does where you just do a bunch of rollouts and compute the correlation between probabilities. This is vLLM vs HuggingFace. 

The rankings in terms of reduced mismatch is
1. vLLM FP16 + Train FP16 
2. vLLM BF16 + Train FP16 
3. vLLM FP16 + Train BF16 
4. vLLM BF16 + Train BF16 

![[Pasted image 20251105095410.png|400]]

More examples of reproductions rain in:

DeepSeek Distill 7B
![[Screenshot 2025-11-05 at 10.05.14.png|600]]

A bunch of ablations with different compile settings, all showing similar mismatch under BF16. Instantly stabilizes under FP16
![[Screenshot 2025-11-05 at 10.05.48.png|600]]


Importantly, the experiments in the paper is performed on **A100**. Interestingly, in the larger-scale experiments that were performed on 64xH100, see bottom row in the first image, there is no training collapse, only slower convergence. In a tweet the authors note that "*We also did not observe severe collapse until we develop the "sanity check" setting - a small dataset with all solvable questions (we can expect 100% rewards in theory and we empirically got 98%).*" 

In the blog post from the Qwen Team [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) that introduced sequence level IS as an unbiased fix for the mismatch (argues that token-level IS as introduced in TIS is biased), they also discussed different hardware and its effects on the mismatch. They found that A100 RL was highly unstable, showing way higher KL when compared to alternate hardware such as L20 and H20. 

@agarwl notes that they experienced no problems when producing results for the scaleRL paper. P
![[Screenshot 2025-11-05 at 15.13.59.png|500]]

This work reproduces the mismatch on H100 atleast. Mismatch and training convergence are not the same thing though! As Samsja points out: does it converge as good as bf16, does it crash at larger scale, we don't know yet.
![[Screenshot 2025-11-05 at 10.21.20.png]]

Some more clear up from Prime coming. The mismatch is persistent on H200 but much lower. BF16 training is stable and running well on H200. General takeaway is that you can probably do BF16 with good algo's but the mismatch will exist, and its impact on training is not completely understood. The examples from the paper are very contrived. 
![[Screenshot 2025-11-05 at 15.23.30.png|500]]

Interesting threads discussing potential differences in A100/H100 that may lead to the discrepency 
https://x.com/danielhanchen/status/1984542983375782098
https://x.com/danielhanchen/status/1984821368291295594

One argument is that the training in BF16 seems fine under normal circumstances. A counter-argument is that the paper shows that these instabilities occur in the long term due to the mistmatch, patching it up with an algorithm fix is great but unclear if this just delays the instability. Obviously we've seen examples of BF16 training be stable for very long terms, e.g the ScaleRL paper, but it might very well be that we are sacrificing performance because higher mismatch is sort of like training off-policy, if we can be more on-policy for free this should be a performance boost.

Trying to decouple the BF16 vs FP16 issue and the hardware differences has seemingly been a bit of a headache. There are again some further insights from the Qwen training-inference mismatch blog post. Apparently, when using something called cascade attention (default enabled in vLLM) on A100s, under certain batch/seq lengths the kernel triggers a path that contains a bug. Disabling cascade attention on A100s reduced vllm-kl by several order of magnitudes. This may very well be the culprit of the hardware related issues, but does the BF16 v FP16 difference persist, we've seen it occur on other hardware as well? According to the authors of the paper, they've already tried disabling cascade attention and still observe the instability under BF16. 

A very solid tip in regards to all of this
![[Screenshot 2025-11-05 at 16.20.45.png|400]]

