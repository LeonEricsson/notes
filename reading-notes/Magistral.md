Mistral reasoning models!! Pretty late on this release tbh. Unfortunately the hype for mistral has decreased **significantly** in the past year. They've fallen behind considerably, chinese labs have taken their place as the non-us competition.  
  
Magistral builds on Mistral Small 3 and Mistral Medium 3, applying a RLVR framework ending up with Magistral version of their latest models.  
  
In line with **every other RLVR work** in the past months, mistral introduce a GRPO variant as their RL algorithm. It's honestly insane I don't know if I've seen a algorithm see this kind of adoption and experimentation in such a short time. Feels like we're getting closer to the end of the GRPO-cycle though, I sense something new on the horizon.  
  
GRPO modifications:  
  
1. Eliminating KL divergence. This has been a bit of a topic of discussion lately, some work claiming that the KL divergence is necessary to avoid entropy collapse. The ProRL paper used KL divergence but reseting the reference policy to a on policy checkpoint throughout training, this seemed smart.  
2. (DAPO) Loss normalization. Normalizing the loss by the total length of the generations to avoid length biases. Nothing new, this has also been discussed extensively in the open research community.  
3. Advantage normalization. Firstly, they remove the division by $\sigma$ (proposed in Dr GRPO paper) because it leads to a question level difficulty bias. Advantage normalization is normal in RL, but in GRPO it is calculate on a question-level as opposed to a batch level, which means that questions with lower standard deviation (too hard or too easy) are given higher weights during policy updates. Mistral propose normalizing the advantages in each minibatch, as $A^{norm} = \frac{(A - A^{mean})}{A^{std}}$ where mean, std advantage are calculate across the advantages in a minibatch.  
4. (DAPO) Clip-Higher, with upper epsilon clipping set to 0.26-0.28.  
5. (DAPO) Dynamic sampling. Filter out groups with zero advantage when forming training batches.  
  
Seems like they've mostly just ripped the DAPO structure straight up (DAPO also removes KL divergence) with a slight modification to advantage normalization following the Dr GRPO paper. This means they must've trained this in just the 1-2 past months.  
  
Their reward shaping is pretty standard, albeit quite naive. They propose formatting rewards for 1) <think> tags, \boxed for math responses, and markdown formatting for code responses. Crucially:  
  
> Failure to meet any of these conditions results in a reward of 0, and the response will not be graded further. Otherwise, the response gets a reward of 0.1 and proceeds to grading.  
  
which seems to have had a negative impact on the model because i've seen several cases where the model responds with mathematical formatting to normal questions. People should probably start using a reward rubric instead of enforcing specific formatting. RL tends to bleed into other domains.  
  
They use the length penalty reward signal from DAPO as well.  
  
#### RL Infra  
  
A distributed RL training system is developed. Adhereing to standard async RL setups we have **Trainers**, **Generators**, **Verifiers**. Since the generators are LLMs performing rollouts, they stand for a significant part of the total compute and time. Additionally, the distribution of work time across training is highly skewed and changes ver the course of training: the longest completions can take up to 5x longer than the shortest. A constant tention is keeping the generations as on-policy as possible, so updating their weights as soon as the Trainer has completed a gradient update, while also operating without waiting for each other or the trainers. This is a fairly classic distributed RL problem setup.  
  
Processing batches sequentially, as is typical in a ML training setup, would guarantee that we are always on-policy: start generators on a batch, wait for all rollouts to complete, update the model weights for both trainers and generators, and repeat. However, this approach leads to idle generators and low pipeline efficiency because completion times vary significantly. Instead, magistral generators run asynchronously, continuously producing at maximum throughput without ever waiting for the trainers. Solutions are constantly gathered from the generators, verified, and used to update the trainers. After the trainers have updated, the new weights are send to the generators via NCCL, without discarding the in-flight sequences being generated. Interestingly they just update the weight mid generation, disregarding the fact that the KV cache becomes outdated.  
  
#### Training and Results  
The goal is to answer two questions (i) how far can one get with pure reinforcement learning on a large base model? (ii) given a strong teacher model, how can one achieve the strongest possible lightweight model.  
  
Magistral Medium was trained from Mistral Medium to answer question (i), and Magistral Small is trained with SFT traces derived from Magistral Medium to answer (ii).  
  
**Magistral Medium**  
Training was done in three stages, ensuring that:  
  
1. Data was not too easy. As the model performance increases, easier data was filtered out and the dataset difficulty was increased  
2. Generation lengths don't stop growing. To prevent stagnation in generation length they adjusted the length penalty.

**Magistral Small**
Three different variants of small are trained, pure SFT, pure RL and SFT -> RL. The SFT is a mix of CoT from Magistral Medium combined with some diverse SFT data. Interestingly the results differ from what DeepSeek found - it is indeed beneficial to perform SFT -> RL as opposed to just RL during knowledge distillation. 

#### Ablations
A couple of fast firing ablation studies, the provide quite little info on these, mostly just results.

**Cross-domain generalization**. Training RL models purely on math and evaluating on math+code, and vice versa shows strong performance to out-of-domain tasks. 

**Distillation vs RL**. The DeepSeek R1 work observed that smaller models benefit more from distillation as opposed to solely relying on RL. Mistral don't observe the same behaviour and find that performance between distilled models and RL is very on par. Likely this is very data dependent, how your SFT data looks vs your RL pipeline, its hard to say what is optimal. Do what's easiest i guess.

**Advantage Normalization**. Mistral observe little differences on downstream performance when comparing advantage normalization in minibatch, group, or none. 

#### Analysis

**Multimodal free lunch** Despite training on purely textual data, performance on multimodal benchmarks improves during RL training.

#### Failed attempts

**Partial reward**. The strict requirements of competitive programming, in terms of correctness and adherence to complexity constraints, result in sparse rewards, often causing many code generations to be discarded due to limited reward diversity. To address this, they experimented with a proportional reward: based on the fraction of tests passed, as opposed to the binary reward. While such training was faster, the resulting model had lower performance.

**Entropy control**. They experimented with entropy loss bonus, but since entropy varies significantly depending on the dataset they found it difficult to determine a general entropy bonus loss term. Instead, setting a higher clip was a more stable approach. Another alternative is to add a KL term to the loss (KL divergence). However, the distribution will differ significantly from the original model during training, they tried using an exponential moving average as the reference for KL but ultimately found it easier to manually adjust epsilon high.