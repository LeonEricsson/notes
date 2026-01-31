## Reinforcement Learning for Reasoning in Large Language Models with One Training Example

vERL framework defines the GRPO loss function as containing three components: *policy gradient loss*, *KL divergence loss* and *entropy loss*.

1. The policy gradient loss encourages the model to produce response sequences with higher rewards, assigning weights to sampled outputs according to their group-normalized advantages. Thus, 3 better-than-average solutions are reinforced, whereas inferior ones are penalized. Since we focus on mathematical problems, the reward is defined as binary.

2. The KL divergence loss measures the divergence between the current model’s responses and those from a reference model, serving as a regularization term to maintain general language quality. 

3. The entropy loss, applied with a negative coefficient, incentivizes higher per-token entropy to encourage exploration and generate more diverse reasoning paths. Entropy loss is not apart of the original GRPO formulation but is included in vERL.

--- 

To find the most "impactful" training examples the authors perform a mock training over $E$ epochs and record the training accuracy for each sample across epochs $0 -> E$. The samples with the highest variance are selected for 1-shot RLVR.

training is done by selecting N samples for N-shot RLVR, typically 1 or 2, and repeating this sample to match a full training batch size.

turn out, RLVR with just 1 example (picked according to historical variance score) performs on par with training on the complete DeepScaleR subset dataset of 1.2k training examples or on MATH train set of 7.5k training examples. Performance is measured on MATH500, AIME202{4,5}, AMC 2023, Minerva Math, OlympiadBench so a fairly robust evaluation set. This is pretty crazy. 

![[Screenshot 2025-05-19 at 15.17.59.png]]

![[Screenshot 2025-05-19 at 15.18.46.png]]

i mean common, really? hahaha, this is honestly pretty funny. to make things even more annoying, generalisation seems to improve as well. Generalisation to non-math reasoning benchmarks ARC-E and ARC-C improves under 1-shot / 2-shot RLVR compared to full training.

![[Screenshot 2025-05-19 at 15.22.50.png]]

what could possibly be the reason for this? one theory about general reasoning models i've heard is that the reason they get better from RL is not because they learn to solve problems better but rather the majority of their improvement comes from implementing a "problem solving scaffolding" to solving a problem. If we imagine that there are certain key aspects for a LLM to solve a mathematical reasoning problem, e.g developing a plan, considering valuable alternatives, backtracking, self reflection. These "scaffolds" can be learnt from a single problem as long as it is difficult enough, because we are only learning the structure to how to solve these problems, that's what is important here. This could explain why training with just a single sample is enough to see such strong improvements.

the authors observe continous test performance improvements during 1-RLVR training long after training accuracy saturation. Training reaches 100% accuracy within 100 steps, but average performance on 6 eval benchmarks continues to improve for 2k steps. Such phenomena is not observed when training multi-example datasets such as DSR-sub. Moreover, at the final stages of 1-shot RLVR, the model overfits the training example by mixing correct calculation process with long unintelligible multilingual outputs in its reasoning trace. despite this the test cases still behave normally and achieve high accuracy.

the authors find that 1-shot RLVR is effective for most type of samples, not just the ones with high base accuracy variance. Improvements are seen in all different domains in MATH500, instead of just the domain the single training example is taken from. In some cases even, the model sees better evaluation performance generalization in domains other than the one the training example is from.

### analysis

1(few)-shot RLVR provides strong evidence for the recently established hypothesis that base models already have strong reasoning capabilities and that we are simply eliciting these capabilities. We've seen work supporting this observation by showing that, with respect to pass@k metrics, models trained via RLVR gradually perform worse as k increases. This work corroborates these findings by showing approaching the problem from a different perspective, that of training a model with guarantee of no new knowledge. 

further ablation studies finds that: Policy Gradient Loss is the Main Contributor, and Entropy Loss Further Improve Post-Saturation Generalization

## Absolute Zero: Reinforced Self-play Reasoning with Zero Data

There has been significant work on improving model reasoning through RL through RLVR where we find a domain, such as coding, maths, where we have a bunch of tasks that have deterministic and scalar-like answers that are easy to verify. This makes it very "easy" to apply large scale RL and get good results. However, such data is limited and there has long been an interest in how do we apply something like self-play to models to make them better by finding an environment where models can play against themselves and self-iterate in a isolated manner without **any** human data whatsoever. We know this paradigm can achieve superhuman performance due to the success of game AI like AlphaZero, AlphaGo, Dota 5 etc but the problem has always been how we define an environment with a verifiable reward signal that translates to real world applications.

### Preliminaries
##### Supervised Fine-Tuning (SFT)
SFT requires datasets of task-rationale-answer demonstrations $D = \{(x, c^{\star}, y^{\star})\}$, where $x$ is the query, $c^{\star}$ is the gold chain-of-thought (CoT), and $y^{\star}$ is the gold answer, all provided by human experts or superior AI models.

The model trains to imitate the reference responses to minimize the conditional negative log-likelihood (Ouyang et al., 2022):

$L_{SFT}(\theta) = - \mathbb{E}_{(x,c^{\star},y^{\star})\sim D} \log \pi_{\theta}(c^{\star}, y^{\star} | x)$.

However, at the frontier level, there’s no stronger model to distill from, and expert human labeling doesn’t scale well.
##### Reinforcement Learning with Verifiable Rewards (RLVR)
To move beyond the limits of pure imitation, RLVR only requires a dataset of task and answer $D = \{(x, y^{\star})\}$, without labeled rationale. RLVR allows the model to generate its own CoT and calculate a verifiable reward with the golden answer $r(y, y^{\star})$. However, the learning task distribution $D$, with its set of queries and gold answers are still labeled by human experts. The trainable policy $\pi_{\theta}$ is optimized to maximize expected reward:

$J_{RLVR}(\theta) = \mathbb{E}_{(x,y^{\star})\sim D, y\sim\pi_{\theta}(\cdot |x)} [r(y, y^{\star})]$.

---

In summary, both SFT and RLVR still rely on human-curated datasets of either queries, demonstrations, or verifiers, which ultimately limit scalability. The **Absolute Zero** paradigm removes this dependency by allowing the model to generate, solve, and learn from its own interactions with the environment entirely through self-play.

### Absolute Zero

The proposed paradigm is described as Absolute Zero, which is a self-play style framework with the following objective 

$$
J(\theta) := \max_{\theta} \mathbb{E}_{z \sim p(z)} \mathbb{E}_{(x,y^{*}) \sim f_e(\cdot|\tau), \tau \sim \pi_{\theta}^{propose}(\cdot|z)} [r_e^{propose}(\tau, \pi_{\theta}^{solve}) + \lambda \mathbb{E}_{y \sim \pi_{\theta}^{solve}(\cdot|x)} [r_e^{solve}(y,y^{*})]].
$$

The core idea is to train a single model, which acts as both a creative "proposer" of new tasks and a diligent "solver" of those same tasks, to maximize a combined reward. This reward system is twofold: the model is rewarded through $r_e^{propose}(\tau, \pi_{\theta}^{solve})$ for proposing tasks ($\tau$) that are appropriately challenging and offer good learning potential for its current abilities. Simultaneously, it's rewarded via $r_e^{solve}(y,y^{*})$ for successfully solving these self-generated tasks ($x$, which is derived from $\tau$ by the environment $f_e$) correctly, with $\lambda$ balancing these two incentives.

This entire process is designed as a self-play loop, as described in the Absolute Zero paradigm, where the model "simultaneously proposes tasks, solves them, and learns from both stages" with "no external data required". The significant innovation here is that the "burden of scaling data" is shifted from human experts "onto the proposer policy $\pi_{\theta}^{propose}$ and the environment e". These two components are responsible for "defining/evolving the learning task distribution". As the model's solver capabilities improve, the learnability reward mechanism naturally encourages the proposer to generate tasks that are more complex or novel, fitting the solver's new skill level. This creates a "self-sustainable training" cycle where the model continuously refines its reasoning by generating and tackling an ever-evolving curriculum of its own making, effectively teaching itself.

### Absolute Zero Reasoner
To verify the Absolute Zero paradigm the authors propose a Absolute Zero Reasoner as a first adoption attempt. Naturally a LLM serves as the proposer and solver, and is trined jointly learning to create tasks that push the boundary of its own reasoning capacity while enhancing its ability to solve them efficiently. 

**What is the proposer conditioned on?** A buffer is initialized to hold previously generated tasks. From this buffer $K$ samples are sampled and used as in-context examples to the proposer when it generates a new task. The design is to show the proposer past examples and prompt it to generate different ones to promote diversity.

**How is the learnability reward calculated?** The solver model $\pi$ is used to estimate learnability through Monte Carlo rollouts, computing the average success rate.

##### Takeaways from training

**Cross-Domain Transfer is More Pronounced for AZR**
A notable observation was the superior cross-domain transfer of reasoning skills with AZR. While expert code models trained with RLVR showed only a modest average increase of 0.65 points in math accuracy, AZR-trained models demonstrated much more substantial gains. Specifically, AZR-Base-7B and AZR-Coder-7B, trained on self-proposed code reasoning tasks, improved their math average scores by 10.9 and 15.2 points, respectively. The takeaway here is that AZR training fosters a much stronger generalized reasoning capability that transfers more effectively across different domains, such as from code reasoning to mathematics.

**Bigger Bases Yield Bigger Gains**
The study indicated that the benefits of AZR training scale with model size. Performance improvements were more significant for larger models: 3B, 7B, and 14B coder models gained +5.7, +10.2, and +13.2 points, respectively. This leads to the takeaway that employing larger base models is advantageous for AZR, as it results in greater enhancements in reasoning performance, hinting that continued scaling is a promising avenue for this training paradigm.

**Comments as Intermediate Plans Emerge Naturally**
An interesting emergent behavior was observed when AZR models tackled code induction tasks: they often interleaved step-by-step plans in the form of comments within the code. This behavior mirrors the ReAct prompting framework and has also been seen in significantly larger formal-math models like DeepSeek Prover v2. The authors believe a key takeaway is that allowing models to utilize intermediate scratch-pads for generating long-form answers could prove beneficial in other domains beyond coding.

**Cognitive Behaviors and Token Length Depend on Reasoning Mode**
AZR training led to the emergence of distinct cognitive behaviors such as step-by-step reasoning, enumeration, and trial-and-error. These different behaviors were particularly evident across different types of tasks. Furthermore, while token counts (response lengths) generally grew during AZR training, the magnitude of this increase also varied by task type. For instance, abduction tasks, where the model often performs trial-and-error until the output matches, showed the most significant growth in token length, whereas deduction and induction tasks grew more modestly. The takeaway is that AZR training not only fosters diverse problem-solving strategies but also shows the model adapting its verbosity and approach based on the specific reasoning mode required by the task.

**Safety Alarms Ringing**
The research also highlighted a potential concern: the AZR model, specifically when implemented with Llama3.1-8b, occasionally produced concerning chains of thought, termed "uh-oh moments". An example of such an instance is shown in Figure 32 of the paper. This finding brings a crucial takeaway: while AZR demonstrates considerable promise in advancing reasoning capabilities, there is a pressing need for future work focused on safety-aware training methods to mitigate and manage these potentially problematic model behaviors.