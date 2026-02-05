### Interplay between pre-training, mid-training and RL

In this paper, the authors show that a RL'ing on the *edge of competence* is key to unlock new capabilities, increasing performance across pass@1->128. The takeaway was that you want to filter your RL data to target tasks where your model fails at pass@1, but succeeds at pass@k. But, to what point is this true? The authors found that when pre-train covered op=2-10, and RL op=11-14, the model strongly improved on 11-14, but also on the 15-20 regime. The base model had a pass@128 of 0% in the 15-20 regime, so, naturally you probably don't need pass@128 to be non zero across entire tasks, a non-uniform distribution is fine, but the model needs some signal to learn, and then get better.

But, if the model is failing at the task, then there's no reason to include it, right? So ideally you want to curriculum learn and introduce tasks once the model sometimes finds a solution. The trade-off here is interesting, and how to balance this. 

Could we fill idle accelerators with such explore samples, that are currently unsolvable but that we want to add to the training data
### RL Synthetic Playground

What algorithms are actually better than another, can we create a controlled playground to properly evaluate algorithms against each other? 

In practice, when people introduce new algorithms they evaluate on undertuned baselines, different base model checkpoints, hyperparameters, optimizer, learning rate, lora settings, prompt templates. RL research is currently missing the equivalent of: standardized tasks + standardized baselines + standardized sweeps + standardized evaluation, so the literature often measures “who built the best end-to-end recipe this month” rather than “which algorithmic idea generalizes.”

### Physics of Language Models

 How do hyper-connections perform in the synthetic playground, specifically when paired with canon layers?

---

In the last result of Part 4.2 ZAZ notes that even after 8B / 1T pre-training on Nemotron-CC , models still fail a simple 2-hop birthday retrieval task. In the video this is used to further motivate synthetic playground, why would we use normal pre-training settings to evaluate architectural differences if a model can't even do 2-hop after pre-training? When does olmo actually learn 2-hop? Olmo supposedly has a bunch of checkpoints released throughout training

### Kernel Programming

https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
https://www.gpumode.com/v2/home

### Tiny Recursive Models 
Interesting research problem where tiny recursive models are able to outperform much larger models on things like ARC-AGI. TRM performs recursion on both a reasoning latent z and a prediction latent y. 
https://arxiv.org/pdf/2510.04871

![[Screenshot 2025-12-02 at 20.52.47.png]]

**ARC is a vision problem**
Paper from a team at MIT who formulate ARC as a image-to-image mapping problem using ViT. Achieves very high performance using models which are very small (6M - 66M).
https://arxiv.org/pdf/2511.14761

![[Screenshot 2025-12-02 at 20.54.54.png]]


### Pipeline RL
Get better at understanding asynchronos RL pipelines. Could we overoptimize performance on a single node? Asynchronos RL, in-place weight updates, continous batching, ... 

Maximize performance on a single node! Go all out to optimize an async framework on a single node