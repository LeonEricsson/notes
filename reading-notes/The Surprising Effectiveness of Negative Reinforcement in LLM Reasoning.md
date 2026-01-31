

This paper decomposes standard RLVR into *Positive* and *Negative Sample Reinforcement*. Using a simplified `+1/-1` outcome-base reward, the RLVR objective is split into two subobjectives, one which only learns from correct responses and the other from incorrect responses. Positive sample reinforcement (PSR) and negative sample reinforcement (NSR). This decomposition allows us to investigate how positive and negative reward signals shape model behaviour. 

A big part of this study is understanding how PSR / NSR training effects reasoning effectiveness and reasoning capacity, formally investigated by looking at pass@k. 

#### Experiment setup

Train Qwen2.5-Math-7B (remember we need to be careful about drawing conclusions from this model) and Qwen3-4B using 4 different RL algorithms: PPO, GRPO, PSR and NSR. MATH dataset containing 7500 problems. Evaluation is performed on MATH, AIME25 and AMC23. An unbiasted estimate of Pass@k is performed with increasing k -> 256.

#### Results from comparing PSR, NSR, GRPO, PPO.

The first results are in the form of pass@k curves on trained models.

![[Screenshot 2025-06-17 at 13.06.20.png]]![[Screenshot 2025-06-17 at 13.06.36.png]]

Note that NSR achieves comparable pass@1 performance to GRPO suggesting that NSR is able to reinforce correct responses indirectly by suppressing incorrect ones.  The authors then claim that NSR outperforms the base model at large k, which one could argue, i feel like the pass@k curves for qwen3 should have been tried at higher k's, just looking at the shape of the base model curves they don't seem near saturated. Generally i feel fine claiming that NSR seems to perform similarly to standard GRPO. 

PSR on the other hand is very underwhelming, ignoring the qwen2.5 math results because this model is cooked, PSR doesn't improve on the base model at all? The authors claim that PSR improves accuracy at the cost of diversity but im inclined to disagree, it seems to me like (based on qwen 3 results) PSR doesn't learn the model shit, it performs the same at pass@1 and worse at pass@(high k). 

#### Why NSR works

Experiments trying to understand why NSR works

**Entropy Collapse**
As we've seen a lot recently the authors look at entropy during training as a way to capture model diversity. Entropy is calculated on a held-out test set. We see that entropy collapses under PSR, slowly declines under GRPO and is held high under NSR. Interestingly, PSR higher 

1. Ratio of correct responses per batch on the training set
2. Ratio of fully-solved prompts per batch (i.e all rollouts are correct).

Indicating the *exploitation* behaviour we've seen so frequently lately when models under RLVR trade greedy decoding accuracy for exploration capability.

**Gradient Analysis**
The authors perform an interesting token-level gradient analysis. They derive the gradient of the loss w.r.t token-level logits at each step, meaning one can observe the gradient directions of each token in the probability distribution at each step. The formulation shows that PSR increases the logits of tokens appearing in correct responses while decreasing the logits of all other tokens, as training progresses it repeatedly amplifies the probability of observed correct sequences. Given our problem formulation, there are many different paths towards the same solution, and PSR will explicitly surpress alternative generations. This is fine if you manage to control entropy during training and make sure that the model explores a wide range of alternatives all the time, but it is very easy to fall in the trap of continual sharpening of the output distribution towards a single way to solve a problem, especially in cases where the same examples may be encountered frequently.   Over time the model's behaviour may collapse into a narrow set of responses. 

In contrast, NSR works by penalizing the logits of tokens in incorrect responses, softly redistributing the probability mass to other candidate tokens. Importantly this increase in other token logits is proportional to their current likelihoods. This has several desirable properties:

1. **Preserving high-confidence priors**: When the model assigns high probability to certain tokens (i.e., πyt → 1) that appear in incorrect outputs (e.g., common grammatical or linguistic constructions), the negative gradient from NSR is scaled by (1 − πyt ), resulting in small updates. This allows NSR to penalize mistakes without erasing fundamental knowledge learned during pretraining.
2. **Prior-guided probability redistribution**: NSR performs a soft reranking of the output distribution by boosting unsampled tokens’ logits zv in proportion to their current probabilities πv . This allows the model to effectively explore and search for better candidates according to their prior beliefs.
3. **Implicit regularization against overfitting**: NSR updates only when the model generates incorrect responses. Once the model consistently avoids these mistakes, NSR naturally halts further updates on those samples. This stopping criterion prevents the model from overfitting or collapsing diversity in examples it has already mastered.

A problem under NSR (that is not an issue under PSR) is of course that partially correct responses are down-weighted. However, the argument is that this is a fair trade off because the main benefit of NSR is that it preserves prior model knowledge and output diversity! Penalizing only incorrect responses can be thought of as pruning the tree paths of wrong answers only, leaving the remaining distribution intact. 

This gradient analysis is performed based on a simple REINFORCE-like objective, extending the analysis to PPO and GRPO the authors find that PPO and GRPO modifications only act to stabilize learning (clipping, advantage, kl reg), the core gradient behaviours identified just above still hold.

#### conclusion

overall interesting paper. good analysis, to the point.