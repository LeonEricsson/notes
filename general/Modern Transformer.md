A rundown of recent popularized changes in the transformer architecture.
## Un-modern Transformer
We assume the following standard transformer as our starting point of exploration

**Transformer block**
RMSNorm
Grouped Query Attention
RMSNorm
FFN SwiGLU

RoPE

You either see pre-norm:
h = x + attn(norm(x))
h = h + mlp(norm(x))

or post norm
h = x + norm(attn(x))
h = h + norm(mlp(x))
or both

Sliding window attention, sometimes alternating with global attention. Dual chunk attention, ser lite olika varianter av local attention för att få attention att bli linjär.

## What's new

### Attention

MLA

Linear Attention

Gated Attention

##### MiniMax
Released M2 just now (Oct 28) and decided to go back from using a hybrid linear attention variant to dense/full attention. This is definitely a move against the current trend, where it seemed like frontier open labs were experimenting more and more with hybrid solutions.  How come?

From MiniMax themselves they say "We are always working on [linear attention]." But training modern LLMs is a jungle and their use cases just keep growing. You constantly have to balance performance across a platitude of domains, considering performance on code, and math, while juggling performance on agentic scenarios. How does it handle multimodality? Can you stable RL the model? The list goes on. One has to remember that linear attention is a reduction in capacity, and for it you have to pay a price. There is a efficiency-performance tradeoff. Given infinite compute we would choose to push softmax attention to its limits, nobody has done this yet, such the race for efficient attention is a race to save compute. 

Building a model that can pratically be deployed and used by the community is a balance of Quality, Speed (TPS) and Price, and in this trio Quality is non-negotiable. Nobody will use a model that is bad, even if its cheap and fast. To understand quality we must evaluate and benchmark, something that is very elusive in this field. We constantly develop new benchmarks which are saturated within a few months once the industry puts attention to them. What you really need is an evaluation system that is comprehensive and actually reflects a model's true capabilities. This is one of the hardest and most important parts of LLM development and becomes even more acute when you start messing with a component as fundamental as attention. This is, according to the MiniMax team themselves, the issue which lead to MiniMax M1, a failure to truly understand the effect of linear attention on *quality*. When the team was building MiniMax-Text-01, everyone was evaluating on MMLU, BBH, MATH, and LongBench (all of which are now saturated). From the perspective of a year ago, with these benchmarks at the forefront, a hybrid attention model looked just as good as pure full attention, and tests with small-scale hybrid models further pushed this misconception on the leaderboards. 

But there was no free lunch, when scaling the model one could see clear deficits in complex, multi-hop reasoning tasks. Naturally, the team developed proxy metrics for this specific weakness and interated until the hybrid model seemed to match MHA, but does the proxy metric correlate to downstream performance? How well does this translate to other tasks? What other weaknesses are missed? It's like a game of whackamole. Being intelligent and rigorous on architectural design is only valuable if you are able to validate your theories on representative benchmarks, and capturing "everything" in these benchmarks is difficult. The better the models get, the harder they are to evaluate. As tasks get harder, the amount of experiment compute required just to get a statistically significant signal on your metric grows astronomically. "You never really know what's going to happen until you scale up". It's obvious that we're still in the very early days of understanding this problem, perhaps the closed labs with longer experience have a better understanding for this? Systematic evaluations of arch design in a way that translates across scales is the outmost problem here, pair that with the fact that training data distributions shift constantly, and architectures behave very differently under different data distributions. 

The infra around Linear Attention is also far less mature than softmax attention.

IV. What’s Next 
"Scaling remains the name of the game, and context scaling is one of the key problems. Longer and longer context length is key in both pre-training and post-training. As GPU compute growth slows while data length keeps increasing, the benefits of linear and sparse attention will gradually emerge. We should start preparing now: Better Data: More multimodal, information-rich long-context data. Better Evaluation: More informative evaluation system and experimental paradigms to speed up iteration. Better Infrastructure: Mature training and inference infrastructure to fully squeeze out GPU potential."

"Partial RoPE very important for long context extension" 
### FFN

MoE

### Other

MTP



