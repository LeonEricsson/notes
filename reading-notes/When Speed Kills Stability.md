Qwen are also acknowledging difficulties in RL stability due to the training inference policy mismatch. 

We've seen many open labs at this point point out this issue which is interesting. It seems most of the open labs have built their own inference framework on-top of vLLM or SGLang which is where we've seen these discrepancies the most. This always makes you wonder how long ago the closed labs came across and solved these issues. It's hard to imagine that you don't run into these issues at all given that training frameworks and inference frameworks optimize for very different performance metrics. 

The [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) regarding batch variant kernels also discussed this, noting that sampling in vLLM and SGLang isn't deterministic. Solving this through batch invariant kernels is one alternative but unfortunately this hampers performance significantly. Maybe the closed labs have already solved this and have fast internal kernels which are invariant.

Anyway, the post notes that by tracking the KL divergence between vLLM and the training framework through a k3 estimator

```python
rollout_log_probs = batch.batch["rollout_log_probs"] # pi_vllm
actor_old_log_probs = batch.batch["old_log_probs"] # pi_fsdp
response_mask = batch.batch["response_mask"]
log_ratio = actor_old_log_probs - rollout_log_probs 
vllm_k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
vllm_k3_kl = masked_mean(vllm_k3_kl_matrix,response_mask)
```

they empirically observe that there is a high correlation between KL and model collapse. These warning signs arise through several different metrics.

We see entropy fluctuations in the FSDP policy align with spikes in vllm-kl. There is less obvious correlation with the rewards, but we can see at step 250 where vllm-kl rises significantly how rewards see a harsh decline

![[Screenshot 2025-10-17 at 08.57.45.png|450]]

Further, the Qwen team observe how both gradient norm and the PPL of the FSDP policy explode in conjunction with vllm-kl. Notably we see how gradient norms and PPL trigger after vllm-kl starts rising, in a delayed fashion, indicating that vllm-kl may be a determinant of modal collapse, acting in a cause-effect relationship. Could it be that increased training-inference mismatch is the root cause of training collapse? Certainly seems like it based on this alone (ignoring that we probably already know that it is)

![[Screenshot 2025-10-17 at 09.01.25.png|450]]

Next, the team analyses the conditions under which the discrepency between the inference policy and FSDP policy are the most extreme. As I would expect, the findings indicate that that as the inference probabilities approach 0, meaning tokens that are sampled with low probability, have a increasing difference from the FSDP probabilities. 

![[Screenshot 2025-10-17 at 09.19.59.png|300]]

In this image they've picked out only the tokens where vllm-kl is already quite high, but the same trend can be observed across all vllm-kl levels, as probabilities approach 0, the mismatch between training and inference increases. 

#### hardware mismatch

finally, the team discover physical hardware as a critical variable. The same code and model produced drastically different levels of mismatch across different GPU hardware. 

![[Screenshot 2025-10-17 at 09.28.27.png|500]]

The differences between accelerators are on the orders of magnitude, with A100 having consistent KL values in the 1e-2 and 1 range, which was so severe that training on A100 became unfeasible.  An interesting consequence of this phenomena is observed when resuming a failed L20 experiment checkpoint on a H20 GPU (which are generally more stable). The training immediately recovered and stabilized

![[Screenshot 2025-10-17 at 09.31.53.png|500]]

#### inaffective solution attempts

**FP32 LM Head**
As far as I know, the first time this training-inference mismatch is mentioned w.r.t RL training is in the Minimax M1 technical report.

![[Screenshot 2025-10-17 at 10.04.09.png|500]]

The proposed fix is increasing the precision of the LM head to FP32. However, the qwen team did not find that this stabilized their training and they were still seeing vllm-kl explosions despite increasing LM head precision.

**Disable Chunked Prefill**
Disabling chunked prefill did not resolve the issue.

**`enforce_eager` and `free_cache_engine`**
The VeRL official recipe of DAPO mentioned that enabling CUDA graphs may cause model performance degredation. To investiage this they conduct ablation studies where these are disabled. Performign a grid search over the boolean `enforce_eager` and `free_cache_engine` hyperparametrs, finding no correlation to changes in training-inference mismatch or improvements on test performance.

#### effective solution attempts

**importance sampling**
The theoretically sound approach when you have a policy used for generating rollouts that differs from the policy being trained is to introduce an importance sampling term. However, the team argues that this IS term should be a sequence level IS to ensure an unbiased policy gradient estimator. That is you apply a single importance ratio over the entire generated sequence (trajectory). This is not in line with the Truncated Importance Sampling that was introduced back in August to mitigate this issue as they used a token-level IS implementation. They validate these findings with experiments, showing that sequence level TIS is able to recover a crashed experiment while token-level TIS fails to. Generally they find that token-level TIS prevents collapse in simpler reasoning environments where the mismatch is smaller but does not work for more complex scenarios.

Further, while TIS prevents complete collapse, the reward curve under TIS exhibit continous fluctuations and the test performance under-performed the vanilla setting

![[Screenshot 2025-10-17 at 10.29.16.png|500]]
*TIS reward curves fluctuate and don't improve over vanilla experiments*

Comparing TIS to MIS, that is masked importance sampling first introduced by the Ling team in IcePop, which masks the loss for sequences where the IS ratio exceeds the threshold C instead of truncating it; MIS not only stabilizes training but also surpasses the peak training rewards of both vanilla and TIS experiments

![[Screenshot 2025-10-17 at 10.30.44.png|500]]
*MIS improves over TIS*

Given that the Ling team and Qwen team are sister labs its not surprising to see them support MIS. Comparing sequence-level and token-level MIS they again find that token-level collapses, reinforcing the conclusion that for complex, long horizon autoregressive tasks, only a theoritcally sound sequence level correction is reliable

![[Screenshot 2025-10-17 at 10.32.16.png|500]]
*Token level MIS crashes, sequence level MIS is sound both in theory and practice.*