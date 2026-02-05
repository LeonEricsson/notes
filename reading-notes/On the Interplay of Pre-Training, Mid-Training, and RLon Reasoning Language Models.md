
The setup is training on GSM-Infinite like dataset. Similar to Physics of Language Models Part 2. Based on a dependency graph G and contextual templates, the authors generate reasoning problems of controlled complexity.

![[Screenshot 2026-02-03 at 10.14.06.png|300]]

The evaluation focuses on reasoning across two axes: explorative (reasoning depth) and contextual (reasoning breadth) generalization. The complexity of the dependency graph is controllable, and it is defined as the number of arithmetic operations required to solve the problem, extractable from the dependency graph.  This is referrred to as `op()` throughout the paper, that is the number of operations required for a problem. The authors argue that breadth generalization comes from being able to transfer the reasoning to novel domains, typically in the form of different templates.

Solutions are verified programatically. Each problem has a gold graph G that was used to generate the problem. Model free-form text solutions are parsed into a predicted dependency graph G' and final answer a'. This dependency graph is then compared, on each node level. The process accuracy is computed as the **average step-level accuracy across all gold nodes**. A prediction is considered fully correct only when both the reasoning steps and the final answer match. All pass@k metrics are reported with respect to this strict criterion.


### When Does Post-Training Incentivize Reasoning Beyond the Base Model?

A model is pre-trained on `op=2-10`, then 4 different post-trained (GRPO) models on ID problems `op=7-10`, OOD-mixed problems `op=9-12`, OOD-edge `op=11-14` and OOD-hard `op=17-20`. Evaluating on three regimes, we see the following:

![[Screenshot 2026-02-03 at 10.24.55.png|600]]

For ID tasks, we see familiar behaviour where pass@1 performance increases, but pass@k>4 remains the same, no matter the training regime. RL seems to sharpen existing capabilities, but it can't extend them. For the OOD tasks however we see something interesting. The models trained on the hard tasks, perform just as well as the base model across all OOD tasks `op=11-20`, we can probably predict this result if we look at the OOD-hard figure where the model scores 0, which means it probably hardly got any signal during training, it never learnt anything because it produced any correct solutions. The RL `op=7-10` model is not able to generalize beyond the training distribution. However, both models which train around the OOD-mid regime see strong improvements across the pass@k spectrum, and even more interesting the RL `op=11-14` is able to generalize beyond depth 14 and show strong improvements in the `op=15-20` range. 

**Takeaway.** RL produces true capability gains (pass@128) beyond base models when two conditions hold: 1) The task is not heavily covered during pre-training, leaving sufficient headroom for exploration and 2) the RL data is calibrated to the model’s edge of competence, neither too easy (in-distribution) nor too hard (out-of-distribution). This provides strong guidelines for RL training. We want RL data to lie on the *edge of competence*. The best achievable results are when we train **exactly** on the edge of competence, `op=11-14`, surpassing the performance of the mixed regime `op=9-12`, and is able to generalize beyond its training. Naturally, you want to filter you RL data to target tasks where your model fails at pass@1 but succeeds at pass@k. You could also employ a curriculum learning scheduler here. 

**Discussion**. One thing to note for this setting was that the base model has a pass@1 = ~10%, and pass@128 = 40% in the `op=11-14` regime. These saw an increase to 50% and 85% respectively when training in this regime. In the `op=15-20` regime, the base model has pass@128 = 0%, but the `op=11-14` model is still able to considerably improve performance in this regime across the entire pass@k spectrum. This likely means that as the model improved in the 11-14 range, it slowly started to produce solutions in the 15-20 regime, going beyond what the base model was ever able to produce, showing true new capability gains. It would be interesting to see the a model trained across `op=11-20` for the same compute, and understand the trade-off between compute and learning signal. You are probably wasting a lot of compute by not solving the higher end problems, so you are likely better to cycle those into the mix as training progresses. 

### How Does Pre-training Exposure Shape Post-Training Generalization?

**Setup**. Model is pretrained on 99.9% context A and 0.1% context B, both encompassing `op=2-20`. Context B is fairly different from context A, being a long-tailed, but of course both share the same underlying reasoning priors (logical-arithmetic reasoning). Post-training is performed on ID tasks `op=2-20`, but with varying levels of context B tasks, from 0 -> 100%. On context A, based on previous results, we expect to see improvements in lower pass@k regimes, without improvements on higher pass@k. But, let's see what happens to context B

![[Screenshot 2026-02-03 at 11.17.00.png|600]]

Expected results for context A. Interestingly, in context B we see improvements across the pass@k spectrum, albeit fairly minor in the higher ends. Remarkably, even with 0% context B exposure, improvement is statistically significant. What this tells us is that when the atomic primitives are shared, post-training can help incentivize generalization across different contexts. So, even though context A sees little improvement, there can still be benefit in ID post training for context generalization. However, this is when pre-training contained data across the full difficulty range for context B. A more interesting situation is when our data for context B is limited to trivial difficulty.

**Setup**. The model is pretrained with context A `op=2-20`, but only context B `op=2`. We vary the ratio of context A vs B in the pre-training mix. Post-training is performed with a 50/50 split between context A/B, spanning `op=2-20`. 

![[Screenshot 2026-02-03 at 11.51.35.png|600]]

Here we see that a minimal exposure of >= 1% is required of context B during pre-training to be able to generalize well across the difficulty spectrum `op=2-20`. 

**Takeaway.** RL incentivizes contextual generalization only when the base model already contains necessary primitives. Without minimal pretraining exposure to a new context, RL cannot generalize. RL can not synthesize capabilities from void; it requires latent "seeds" to amplify. However, these seeds need not be complex. In practice this means pre-training should prioritize broad coverage of basic domain knowledge, rules and skills.

### How Does Mid-Training Interact with Post-Training?

**Setup** Starting from a base model trained for 10B tokens on `op=2-10`. They train 5 different model configurations, varying the ratio of mid-training and RL from 0% to 100%. Mid-training here is just continued pre-training on 1B tokens from the `op=11-14` range. RL is on the same op range.

![[Screenshot 2026-02-03 at 12.16.52.png|650]]

On OOD-edge tasks, meaning in distribution for the post-training, but on the edge of the base model capabilities, lower ratios of RL has better performance across the difficulty spectrum. Light RL achieves the best pass@1 performance, and full mid training the best pass@128 performance. For OOD-hard tasks, reallocating more budget toward heavy RL substantially improves performance on the hardest instances in both pass@1 and pass@128.

**Takeaway.** How do deal with this may be difficult, RL seems to be indispensable for generalization to harder tasks, but at the same time mid-training allocation greatly improves this generalization by instilling priors that RL can exploit. Across all settings, Heavy-RL (80% RL / 20% mid-training) beats Full RL. When prioritizing in distribution performance, mid-training with only light-rl seems best. for ood generalization, heavy rl is best.

**Discussion**. This does however assume we can generate high quality mid-training data. In this paper, from what I can tell, the mid-training data perfectly captures the evaluated task, this may not always be applicable.

