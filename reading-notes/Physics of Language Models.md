
Evaluating architectural differences in models is often plagued by confounding factors that diminish and don't truly show the differences in the tested architectures. The paper identifies three distinct challenges in understanding effective neural architecture design

**Challenge 1: Pretraining loss as an unreliable proxy for intelligence**. Arch comparisons often rely on perplexity or cross-entropy loss, but these metrics do not reliable reflect real-world capabilities. Pretraining loss are good at establishing scaling laws but they do not necessarily reflect downstream performance. This has been seen time and time again with for example Mamba models frequently achieving lower perplexity early in training due to rapid memorization, yet perform poorly on complex reasoning tasks. This discussion is actually why I decided to read this paper after holding of on it for several months; MiniMax decided to abandon their linear attention variant after realising it hurt performance in multi-hop reasoning. 

![[Pasted image 20251028123606.png]]

This begs the question of how diligent labs are about expanding pre-training experiments beyond perplexity.

**Challenge 2: Noise below emergence thresholds**. Emergent abilities complicate architectural comparisons at smaller scales. Small benchmark gains are often the result from random intialization or data shuffling that can cause 2-4% swings in accuracy. 

**Challenge 3: Grokking, Data Quality and Curriculum Learning**. Failures in complex reasoning tasks typically stem from deficiencies in training data, not architectural limitations. 

In this version of PLM, the authors attempt to overcome noise and cost of real-world pretraining by decomposing intelligence into atomic components such as reasoning depth and breadth, and design synthetic, controllable pretrain tasks to isolate and evaluate them independently. This approach addresses the above challenges by enabling single-skill evaluations, minimizing the confounding factors prevalent in real-world pretraining data; lowers resources for rigorous comparisons as synthetic benchmarks yield infinite high-quality data where capabilities like deep multi-hop reasoning emerges clearly and reliably. 

I am a strong believer in this approach, I would expect pre-training teams to develop internal task datasets that mimic this kind of isolating behaviour, but that expand beyond the different properties proposed in this paper.

### Synthetic Tasks for Decomposing Intelligence

The first step is to design said synthetic tasks to systematically evaluate specific capabilities of language model architectures under controlled conditions. This approach enables pragmatic, clean, isolated analysis of architecture design choices. 

- **DEPO (Mental reasoning depth):** This task evaluates a model's ability to perform multi-step computation internally, without using Chain-of-Thought (CoT). It is structured as a $k$-hop traversal over directed permutations given in a random order. The model must compute the $k$-th successor for a given query node based on a list of directed edges (e.g., $x_i \rightarrow y_i$). The task is designed to test architectural scalability and efficiency in hierarchical reasoning.
    
- **BREVO (Mental reasoning breadth):** BREVO assesses a model's capacity to process multiple dependencies at the same time, which is necessary for tasks like traversing dependency graphs. The task involves the recursive traversal of directed acyclic graphs (DAGs). Given a query vertex, the model must identify all vertices it recursively depends on and list them in topological order. This structure forces the model to mentally process the entire graph structure before generating an answer.
    
- **CAPO (Knowledge capacity):** This task evaluates how efficiently a model encodes factual knowledge directly into its parameters, measured in bits per parameter. The dataset consists of synthetic biographies containing various attributes (e.g., birthdate, employer). These facts are presented in many paraphrased formats to discourage surface-level memorization. The evaluation is conducted in an undertrained regime, with each biography shown only 100 times, to better expose architectural differences in storage efficiency.
    
- **MANO (Knowledge manipulation):** MANO assesses a model's ability to retrieve factual knowledge embedded in its parameters and then perform internal, hierarchical computation on that knowledge. The task uses synthetic modular arithmetic expressions (e.g., operations mod 23). Models are required to solve these multi-step arithmetic problems entirely mentally, without CoT. This combines the challenge of retrieving stored facts (the operation tables) with in-memory reasoning (composing operations).
    
- **LANO (Hierarchical language structure):** This task tests structural reasoning, specifically the ability to handle hierarchical relationships and long-range dependencies. It uses synthetic datasets generated from context-free grammars (CFGs). The CFGs are intentionally designed with local ambiguity, forcing the model to infer implicit recursive structures across the entire sequence rather than relying on local cues. Resolving this ambiguity requires a global, dynamic programming-like process to map the sequence to a valid CFG parse tree.

#### Initial experiments
This study covers architectures coming from three major families, *Quadratic-time attention* models pioneered by the original Transformer is represented by a Llama-style model. *Linear-time attention* models are represented by Gated Linear Attention arch. *Recurrent and state-space models* are represented by Mamba2. Model sizes are standardized and evaluations are performed across a sweep of sizes, other learning settings (batch size, training steps and learning rate choices) are identical across architectures.

On the synthetic tasks, linear attention GLA performs weakest overall, Mamba2 excels in knowledge tasks (CAPO, MNO) and LLama(RoPE) performs best on reasoning tasks (DEPO, BREVO, LANO). However, the authors note that the while initial results show stark differences, they avoid deeper interpretation at this point. There are however some valuable takeaways:

- Randomness may affect outcomes. Despite multiple seeds and four learning rates per configuration, smaller models sometimes outperform larger ones. Thus, robust statistical comparisons are crucial. 
- Synthetic tasks clearly highlight architectural differences (e.g 90% vs 5%), exposing strengths / weaknesses. This is in contrast to modest differences in typical pretraining metrics. 
- If a specific architecture (of a given size) fails at a certain difficulty level (e.g., large N or k), it does not imply the model cannot learn the skill given infinite training. The comparison uses a fixed, limited training budget: all architectures train for the same number of steps with identical data and shuffling, reporting best accuracy across multiple learning rates. Thus, results should be seen as differences in the speed of skill acquisition, not absolute capability

#### Canon Layers

The paper introduces Canon layers as a method for enhancing horizontal information flow outside of attention. Simple tasks like token recall require careful mixing of local context—not to say more complex ones or when words span multiple tokens. Since MLP layers don’t mix tokens, attention must handle all communication. Canon layers enhance information flow across neighbouring tokens (typically 4), by aggregating nearby hidden states into the current position. Formally the hidden state $h_t'$ for a token at position $t$:

$h_{t}^{\prime}=w_{0}\odot h_{t}+w_{1}\odot h_{t-1}+w_{2}\odot h_{t-2}+w_{3}\odot h_{t-3}$

where $w_0, w_1, w_2,$ and $w_3$ are the trainable weights (the "kernel") that are applied to the current token ($h_t$) and the three previous tokens ($h_{t-1}, h_{t-2}, h_{t-3}$).

The Canon layers are implemented with residual connections. 

The paper argues that this mechanism greatly improves performance of both Llama(RoPE) and Llama(NoPE) on the synthetic benchmarks.

#### MLP and MoEs

Gated MLPs (e.g SwiGLU) show slightly better performance on reasoning-heavy tasks but reduces knowledge capacity by about 30%. Adding Canon layers partially mitigates this capacity loss.

Mixture-of-Experts enhance parameter efficiency and improve inference time performance. However, it suffers from significantly lower knowledge acquisition speed during training. Canon layers partly recover this.

#### Linear Attention + Canon

Linear attention models reduce compute by compressing sequences into fixed-length representations, gaining popularity for their scalability and efficient handling of long contexts.
GLA, like most linear attention models, compresses past tokens via a (gated) averaging mechanism. While efficient, gated averaging often diminishes the influence of nearby tokens—crucial for nearly all tasks. Adding full Canon layers substantially improves GLA performance across all benchmarks, transforming it from a weak baseline into a strong competitor. 

#### Final Comparison

With full-score Canon layers added, the authors find the following rankings between architectures on the decomposed tasks. 

- **Reasoning depth:** $ROPE(\iota) > NOPE > Mamba2 \approx GLA$ 
- **Reasoning breadth:** $RoPE(\iota) \ge NoPE \approx Mamba2 \approx GLA$ 3
- **Knowledge capacity:** $Mamba2 \approx GLA > RoPE(\iota) \approx NoPE$ 4
- **Knowledge manipulation:** $Mamba2 \ge RoPE(\iota) > NoPE > GLA$ 5
- **Hierarchical structure:** $RoPE(\iota) > NoPE \approx Mamba2 \approx GLA$ 6

Reasoning depth remains the weakest point of Mama and GLA, and potentially most linear-time models on the market. This is not a limitation of insufficient recurrent memory. The tested models have more than enough capacity to store the entire input sequence. It is rather tied to memory dynamics: how efficiently in-context information is encoded during compression and how reliably it is retrieved for reasoning. Until linear architectures overcome these limitations, hybrid approaches remain the best practical solution. 

#### Real-Life Experiments

Finally, real-life experiments are performed at an academic scale, pretraining 1.3B-parameter language models on 100B tokens using the FineWeb-Edu and SlimPajama datasets. Across a wide evaluation harness, model performance varies significantly due to random seeds, with fluctuations up to 9%! Hence, only differences exceeding these thresholds are considered statistically meaningful. In *generative evaluation tasks* linear models substantially underperform full transformers. Both GLA and Llama(NoPE) show poor performance in their base configs, but improve significantly with full Canon layers. GLA+Canon often matches or surpasses Mamba2. Llama(RoPE), Llama(RoPE + Canon) and Llama(NoPE+Canon) generally excel across tasks without notable differences.