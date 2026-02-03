#### ideas

- i'd be interested to see how hyper-connections perform in the synthetic playground, specifically when paired with canon layers.
- how does something like olmo perform on the 2-hop birthday retrieval task. at what point is 2 hop-retrieval solved?

## part 4.1a


## part 4.1b

Canon layers introduce horizontal information flow across tokens in a transformer. In a standard transformer, the only horizontal information flow is in attention. Canon is a weighted sum of the three previous token positions. You can either use random fixed weights w0, w1, w2, w3, or trainable weights which is equivalent to a Conv1D layer. You want add this as a residual, to retain the residual stream: h´ = h + casual_conv1d(h). There are 4 reasonable positions in a transformer block: pre attention block, after q/k/v position, post attention / pre mlp, post mlp. You want to add Canon layers at as many points as possible. 

NoPE completely fails the synthetic playground, but adding Cannon layers transform NoPE layers to a strong contender to RoPE. This is very interesting because RoPE are a known bottleneck for long context. RoPE is also a problem for short context, it can hurt performance for short context. Because for many short context tasks, the positions of the tokens don't matter, but we are forcing the model to use positions and learn positions. 

1/4 RoPE + 3/4 NoPE + Canon performs best in the synthetic playground.

Across linear models (GLA, GDN, Mamba) adding full Canon improves performance holistically in the synthetic playground. When we compare architectural differences and non-Canon vs Canon next to eachother

![[Screenshot 2026-02-03 at 22.27.26.png|700]]

We see that the intra-architectural differences are outshined by the horizontal mixing differences. The best performance achievable is actually with the simples linear model GLA + Canon. This indicates that the current design direction of linear-model architectures may warrant re-evaluation. This result will be further re-inforced once we take a look at large-scale retraining.

#### result 10 - transformers vs linear models

The biggest fault of linear models is their reasoning depth. Compared to Transformers they fall far behind in their ability to reason deep. 

![[Screenshot 2026-02-03 at 22.32.49.png|600]]

Transformers have a 4x reasoning depth. This discrepancy is not due to Linear models recurrent memory capacity. In this task, the recurrent memory far exceeds the necessary memory to hold the entire task context in memory. It has about 100x more memory than required. The problem actually has to do with linear models learning 1 hop way too slow compared to Transformers:

![[Screenshot 2026-02-03 at 22.36.22.png|500]]           ![[Screenshot 2026-02-03 at 22.36.58.png|500]]

Linear models are very inefficient at compression & retrieval. Until this is fixed, hybrid models will remain essential.
## part 4.2 

This part looks at what happens when we scale beyond the academic pre-training. Many differences and capabilities are not observable at the academic pre-training level, you need to conduct larger scale experiments to observe the differences in standard benchmarks. Canon layers improved performance in the proposed synthetic playground, if we can see the same results transfer to large scale training, this provides strong evidence that ablations and architectural research doesn't need standard pre-training datasets to evaluate, and further, it would argue that using pretraining dataasets to evaluate these differences is out-right wrong, and risks drawing the wrong conclusions.

A slight recap: Architectural differences at academic pretraining scales (1.3B, 100B tokens) can be mostly attributed to noise. We find that architectural differences account for roughly 1-2% variance across benchmarks, while changing something like the random seed can account for up to 4% difference.


### scaling pre-training pillars

##### dataset quality
So, to scale to large-scale models that are equivalent to published models in the space, we need a pre-training dataset. The standard pre-training dataset was something like SlimPajama for a long time, then we improved datafiltering and that resulted in datasets such as FineWeb, and DCLM which all reduced necessary training time. Recent pretraining dataset work has started employing rewriting/rephrasing as a tool to improve knowledge storing. This is in line with results from Part 3 of the PLM series which showed that rewriting improves knowledge storing format. Without rewriting, the model could see the knowledge 100x times but was not able to manipulate the knowledge. Naturally, rewritings occur in crawled pretraining datasets, but training datasets like Nemotron-CC employ rewriting systematically. Rewriting is orthogonal to datafiltering, you want to perform both.

##### train longer
The experiments are run on 1.3B/3B/8B models that have been trained for 1-2T tokens, repeated across **2-3 best LRs**. This is in order to supress noise and properly reveal architecture separation.

**learning rate tuning**
One of the most important learnings in architectural design is: **do not trust other peoples hyperparameters**.
Learning rate is **the** highest order bit when it comes to performance, it has an extremely strong influence on the downstream performance. When comparing things like architectures, optimizers, etc, just picking the baseline hyperparameters from original paper greatly risks comparing to an undertuned baseline. You've almost definately modified the setting compared to the original paper: data, batch size, model implementation; you **have to tune the learning rate**. This is a huge problem, in architectural research, optimizer research, and especially in RL. 

[Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/pdf/2509.02046) is a great poster child for this. They disect years of optimizer research by training models across 4 scales (0.1B - 1.2B) on 1-8x chinchilla optimal data ratios, rigorously tuning hyperparameters for each configuration. They find that optimal hyperparams for one may be suboptimal for another (shocker), making blind hyperparameter transfers illegitimate, but more importantly, they find that the actual speedup over well-tuned baselines turn out to be much lower than expected (i wonder why academic researchers weren't tuning baselines :>). 

Relevant tweets from Lucas Beyer:
![[Screenshot 2026-02-03 at 13.56.59.png|350]]

If you have ever heard that "ideas from most papers don't end up working at scale / in production / long term", this is basically the reason. **Always sweep learning rate**

ZAZ uses "2-3 best LRs" to supress noise across experiments. He says that he prefers to use results across different LRs as opposed to across different seeds. This tests both the models robustness, but also sees if a certain LR is better/worse.

##### tighten experimental controls

This is another thing that often goes amiss in architectural research. This means normal things like same data, data shuffling, tokenizer, but it goes even further into things like the same model depth and hidden size, as well as the same code pipeline. In this series we compare quite different models: dense transformers (Llama), Gated Linear Attention (GLA), Gated DeltaNet (GDN) and Mamba. So we also need to make sure that these **models are maximally aligned**. For transformers, this is pretty easy because we already have standardized model sizes:

| Model Size | Layers | Hidden Dim (d) | Attention + MLP |
| ---------: | :----: | :------------: | --------------- |
|       1.3B |   24   |      2048      | 4d² + 8d²       |
|         3B |   26   |      3072      | 4d² + 8d²       |
|         8B |   32   |      4096      | 2.5d² + 10.5d²  |
*To clarify the last column: For each transformer block, you spend 4d² of trainable parameters on the attention side, and 8d² parameter on the gated (swish) MLP side.*

But, comparing this to linear attention models is difficult. The default configurations are:

|  Model | Layers | Attention + MLP | Recurrent Memory |
| -----: | :----: | --------------- | ---------------- |
| Mamba2 |   48   | 6d² + 0d²       | 24 x 512d        |
|    GLA |   24   | 4d² + 8d²       | 24 x 256d        |
|    GDN |   24   | 6d² + 6d²       | 24 x 192d        |

If you just naively take the default configuration when comparing linear attention models, you've already got a large discrepancy in the configuration, so how do you know if model improvements are coming from your linear attention, or just e.g attention/mlp param split. But. more importantly than this is the **recurrent memory** used by linear models. All linear models have some form of recurrent memory to hold state information across the time dimension. There is also head_dimension, q.k.v head dims etc. Only after removing these hidden confounders can we make fair and scientific comparisons.

### results

#### result 1
According the synthetic playground, Canon layers improve Llama (standard dense transformer). However at the academic pretraining scale, this is not visible in the typical benchmarks. The performance is no better than noise
![[Screenshot 2026-02-03 at 14.28.39.png|700]]

Scaling up to "proper" pre-training scales, we now see statistical significance in Canon improvements. This reinforces two points: 1) horizontal time mixing is beneficial and models improve with it 2) pretraining is noisy, and a poor way to evaluate architectural modifications. These runs cost in the tens of thousands. Running ablation studies at this scale is unfeasible for many but a few frontier labs. The synthetic playground told us that Canon layers is better, but we had to go to 1e22 FLOPs to see this.
![[Screenshot 2026-02-03 at 14.28.27.png|700]]

Just to re-iterate. This is no longer a toy setting like the previous parts. We're not at comparable compute to many open-source models. Also, note the Nemotron-8B-Nemo-1T model, that is exactly the same model as our Llama configuration, just with different hyperparameters. This again re-iterates the point of hyperparameter tuning and how big of a difference it can make. You can see the Nvidia models here are from the Nemotron-CC paper and that they use it to show the improvement in performance at 1T tokens. If proper tuning can get the Nemo configuration up to 62.5, it makes you wonder properly tuned variants of this plot would look like. You can also see that Canon confidently shows gains at large scale.
![[Screenshot 2026-02-03 at 14.36.09.png|600]]

#### result 2 - read the benchmark curves

This is a tip that applies in general, but eval curves are a lot more telling than final tables. We as humans have a bias towards nice clean numbers, we really want to look at a single number and say X is better than Y. This bias leads to things like looking at benchmark averages (meaning average across benchmarks) when two different benchmarks are completely uncorrelated, benchmark averages are a bad thing, we should be forced to look at each benchmark performance individually, but we don't want that as humans. The same goes for benchmark curves, they can be a lot more telling than the final reported number. Maybe you happend to stop and select a checkpoint at a poor position, curves tell the full story

![[Screenshot 2026-02-03 at 15.13.04.png|600]]
*Example of LlamaCanon vs Llama on MMLU. Note that LlamaCanon clearly outperforms Llama, although several bumps which may hide this difference exists. Some benchmarks are better than others, HellaSwag for example has a lot of samples, and thus the curves are really clean. Arc-challenge is very noisy. Use moving averages for such noisy cases*

The curves also provide a nice way to read out the efficiency. Looking at the 8B for example we see that Canon saves ~30% pretraining tokens for the same accuracy target. The final reported number may only be a few percentage points, but the curves show how much these points cost.

Just to re-iterate how much of a save this is. In the Fantastic Pretraining Optimizers paper from earlier they find that modern optimizers (e.g Moun) speedup is reduced from 40% to 10% when scaled up to 1.2B/100B. Our results here are 30x that scale and we're seeing a 30% gain.

#### result 3

Redesign GLA -> GLA5 and GDN -> GDN2 to both improve performance and align the models carefully with respect to MLP size, q/k/v dims, recurrent memory = 256d. This minimizes differences between the model architectures and allows us to truly compare their differences. Naturally we still make sure to use the same data, random seed, random shuffle, total # of tokens, tokenizer, code impl.

#### result 4

Linear attention + Canon performs well. We use Full Canon (Canon-AbCD) to provide maximal horizontal time mixing. GLA5+Canon outperforms GLA5 (same goes for GLA). Saves about 20% tokens on MMLU for 8B/1T. Among the linear models it turns out that, when looking at pretraining benchmarks (lm_eval) **GLA5+Canon ≈ GDN2+Canon**. So Canon manages to lift linear attention to GDN-level performance, funnily enough the training curves are eerily similar. In the figure below, dotted lines are GLA5 and GDN2, while the solid lines are +Canon respectively.

![[Screenshot 2026-02-03 at 16.00.46.png|600]]

Notice how the +Canon lines are almost exactly the same, suspiciously similar. ZAZ did double check the runs and its true, their almost exactly the same. This is the result of very careful model alignment, where we've reduced model differences to the point of only the attention mechanism. When you do this, the models perform the same. So, it turns out that when you introduce Canon layers, it doesn't really matter if we're doing GLA or GDN. Even though GDN is a very smart design, it adds a a forgetting mechanism to your recurrent memory where the model can selectively forget parts of its memory. In theory this is very nice, but in deep learning, the optimizer doesn't care about beauty. Maybe the forgetting mechanism of GDN isn't doing something smart as we hoped, just simple time mixing. Research is rough in that sense, it can be easy to attribute gains to beautiful work for the sake of beauty. 

Without Canon, GDN2 / Mamba2 both beat GLA5, as commonly observed
With Canon, GLA5 ≈ GDN2 > Mamba2.
-> so perhaps most of GDN and Mamba2 advantages come from horizontal flow, not architecture complexity.

#### result 6 - Linear models fail at in-context retrieval

Linear models can outperform transformer models on lm_eval, this was observed in result 4, but they fail on generative + retrieval tasks **even after 8B/1T** heavy pretraining...

ZAZ designs a simple retrieval task:
Randomly generate 5 sentences following the template `<Name> was born in the year 19xx`, along with a question `Answer me: <Name> was born in the year if _`, then bury this in a random wikipedia doc of length L.

![[Screenshot 2026-02-03 at 16.14.44.png|450]]

Transformers blow linear models out of the water, across all context lengths. Linear models are very poor at this task. Look at the performance for L=0, even without any surrounding context, just the 5 sentences and the question, linear models fall down several percentage points on the trivial task. This is after 1T tokens of pretraining btw. At longer context of 4k, linear model collapse to random guessing (20%), this is despite being trained on 16k context. Linear models are no solutions to long-context tasks.

This was actually already pointed out in part 4.1b, where we observed these models in the synthetic playground. Looking at the reasoning depth task, we see that linear models learn 1-hop way slower than transformers. Remember that 1-hop is exactly information retrieval. 

![[Screenshot 2026-02-03 at 16.19.21.png|500]]     ![[Screenshot 2026-02-03 at 16.20.27.png|450]]

Transformers k=16 hop curves looks like the linear models k=1 hop curve. Linear models are terrible at compressing and retrieving. Remember that this is **not** a data problem. In the synthetic playground, the model has the perfect data, pure reasoning data with no noise, and it's still incredibly slow. 

Until this problem for linear models is solved, hybrid models are the only viable solution.
This is the secret behind why these industri hybrid models work: Falcon-H1, Qwen Next, Kimi Linear

#### result 8 - 2 hop reasoning does not emerge, evan at 8B 1T pretraining

If we take the birthday retrieval task from the previous result, and we expand the task by making it into a 2-hop task by introducing sentences like `<Name> was born the same year as <Name>`, then bury this into wiki doc at length L, we find that even at L=0, the accuracy is no better than random guessing (~33%). This applies to **all architectures**. 

This result goes back to again reinforce one of the largest takeaways from this entire series:

**Lesson 1** It's uninformative to study architectural design (or even optimizer design) in real-life pretraining. Even at 8B/1T scale. If the models can't even do 2-hop reasoning, what are we actually comparing? Are you going to waste hundreds of thousands of dollars on pretraining scale ablations when models don't even develop difficult skills at this stage? Pretraining of this size maybe helps you compare knowledge capacity and information retrieval - nothing more.

**Lesson 2** Modern LLMs introduce reasoning data too late; the results form this series suggest multi-hop supervision must appear much earlier. Standard practice today is that multi-hop data is not introduced until the mid-training stage. ZAZ advocates inserting such data a lot earlier in the pretraining stage. Earlier results from the PLM series suggests that certain skills should be learned early in the training, they can't be properly learned during lora/sft. Now, this is obviously not IOC level problems, but simple reasoning problems should be properly injected earlier into the pretraining data.





