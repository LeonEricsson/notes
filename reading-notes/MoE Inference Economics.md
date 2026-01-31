Onto the second post in this series, this time on Mixture of Experts. This has become the defacto standard (large) model in 2025. 

This post thoroughly investigates DeepSeek V3.1 as the MoE architecture of choice.

The key challenge in MoE inference is that, unlike dense models like the previously discussed Llama 3, each token only activates a subset of parameters. In dense model inference, we established that increasing batch size as key to good economics, because the cost of loading model parameters is amortized across the batch. For. MoEs this becomes a lot more challenging, as each request (element in batch) will be routed to a different subset of the model parameters. The experts are chosen semi-stochasitcally, with some tokens routing to the same experts. As batch size increases, more experts will be shared by different requests.  This means that at the larger batch sizes we will partially recreate the situation from the dense model - sharing the cost of model loading between multiple users. Unfortunately this means that we will need significantly more requests, i.e. more users, to achieve the same "economies of scale" for MoE models.

Large MoEs like V3.1, with 670B parameters, automatically force us to go cross-node. And, with the additional requirement of a very large batch size, the KV cache is going to be a bigger pain point, further increasing our total memory demand. A new type of parallelism has emerged as a result of MoEs called Expert Parallelism (EP), we'll talk about this more later but basically we want to split the model so that each GPU handles some subset of experts and we route all relevant tokens to this GPU, this way we don't have to communicate intermediate results as in TP setups. 

Some qoutes from DeepSeek:

	Due to the large number of experts in DeepSeek-V3/R1—where only 8 out of 256 experts per layer are activated—the model’s high sparsity necessitates an extremely large overall batch size. This ensures sufficient batch size per expert, enabling higher throughput and lower latency. Large-scale cross-node EP is essential.
	
	As we have adopted prefill-decode disaggregation architecture, we employ different degrees of parallelisms during the prefill and decode phases:
	
	Prefilling Phase [Routed Expert EP32, MLA/Shared Expert DP32]: Each deployment unit **spans 4 nodes** with 32 redundant routed experts, where each GPU handles 9 routed experts and 1 shared expert.
	
	Decoding Phase [Routed Expert EP144, MLA/Shared Expert DP144]: Each deployment unit **spans 18** nodes with 32 redundant routed experts, where each GPU manages 2 routed experts and 1 shared expert.

### DeepSeek MoE Architecture

There isn't much worth saying about the architecture, its got 256 experts with 8 active and 1 shared. There are 58 MoE layers. MLA is the one thing that really stands out as making a difference. Thanks to the KV latent compression our KV cache has a significantly lower memory footprint. In V3.1, our per token KV cache size is 70KB, compared to something like 516KV in LLaMa 405B. This is a great feature given the memory bound nature of decoding.

The MoE layer as we said is 256 experts (FFN / MLP with SwiGLU) that is essentially the large FFN broken down to a bunch of small parallel FFNs instead. How does this effect compute? Well in the standard dense model, the FFN hidden dimension is typically very large, 3.5x larger than the hidden model dimension (28672 vs 8192), but the expert intermediate size is only 2048. Routing through 8 of these experts the FLOPs are equivalent to a single FFN with a hidden dimension of 16384. 

### Inference optimization techniques
DeepSeek were forced to go to serious optimization techniques due to existing import restrictions limiting their hardware. Let's look at a few of this in more detail.

#### Expert Parallelism
