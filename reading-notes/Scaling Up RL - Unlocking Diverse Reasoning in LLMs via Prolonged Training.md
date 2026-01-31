This is a follow up paper to ProRL, released by NVIDIA last month. ProRL was focused on how to achieve stable RL during prolonged training. This paper seems to carry the torch onwards, by investigating similar questions but in a wider set of verifiable domains. Where ProRL was math/code only, this paper collets more diverse training data spanning maths, code, logical puzzles, STEM-related problem solving, and complex instruction following.

**Data**

Math - 40k math problems sourced from community curated datasets made available through DeepScaleR
Code - Publicly available RL datasets comprising 24k coding problems from various programming competitions.
STEM - SCP-116K, containing 274k scientific problem-solution pairs spanning diverse fields such as physics, chemistry, biology, and mathematics.
Logical Puzzles - Utilizing the Reasoning Gym project, the authors curate a dataset containing 37k synthetic training samples spanning 96 tasks.
Instruction Following - A synthetic generated dataset formated similar to IFEval. For instance, a prompt may ask the model to “Write an essay about machine learning”, while the instruction specifies, “Your response should have three paragraphs.”

#### Approach

Same as ProRL, this paper builds on GRPO and specifically DAPO, attempting to dissect what is necessary for stable training in extended duration.

*Mitigating Entropy Collapse* is crucial for stable prolonged training, because it means the model is capable of exploring new solutions to problems. GRPO specifically is heavily dependent on a diverse set of sampled outputs to its questions, to be able to estimate relative advantage. 

**Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)**
Several components of DAPO are incorporated to address entropy collapse, first, decoupled clipping, and second dynamic sampling.

**KL Regularization**
Just as in ProRL, the authors argue against the removal of KL divergence, and instead incorporate it as a penalty between the current policy and the reference policy using an unbiased estimator. They argue that when starting from a well-initialized checkpoint (e.g one that has undergone SFT already) such as DeepSeek-R1-Distill-Qwen-1.5B, keeping a KL penalty provides stability and sustained entropy, and it is not as big of a problem because the models are already capable of generating coherent CoTs. If you start from a base model, KL reg may be less beneficial because the desired behaviour is further from the policy.

Additionally, the authors observe that the KL term may increasingly dominate the loss as training progresses, and to mitigate this they implement period resets of the reference policy, updating it to the current policy.

#### Ablations

**Rollout Temperature** is important to ensure stable training. They experiment with temperatures varying from 0.6 - 1.2 through different stages of training, finding that high rollout temps (1.2) are preferred both early and late stage training.

**Decoupled Clip Coefficients**.  $\epsilon_{low}$ = 0.2 and $\epsilon_{high}$ = 0.4 performed best. A lower $\epsilon_{low}$ fails to improve training at all, likely because it does not properly down-weight actions with negative advantage. Interestingly however, it did lead to a higher (stable) policy entropy. This is because it effectively flattens the action distribution and encouraging broader exploration. 

**Dynamic Sampling** slightly improves per step validation score as it increases the reward signal density in each batch. However, they do not supply a wall-time comparison which would have been interesting to see given that dynamic sampling requires resampling until the training batch is filled.

**Reset reference policy** seems crucial to ensure long stable improvement throughout training when employing KL divergence penalty.




