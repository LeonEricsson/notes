
#### Scaling with input length
During prefill we have the classic O(N^2) compute scaling due to attention. Decoding, we as established, does not exhibit this and only grows O(N) with sequence length. The consequence of this is that our TTFT outgrows our token-by-token time, and as input length increases a larger and larger percentage of the total time is spent prefilling.

During decoding, the relationship between generation speed and sequence length is less straightforward. FLOPs wise, we scale linearly; with each added token our attention increases, as we established we have O(S) during decoding. However, FLOPs are not that relevant because we are memory bound. On the memory side, we have a constant cost represented by the model parameters, and a dynamic cost in the KV cache which scales linearly with sequence length. Initially, model parameter loading will dominate KV cache but as generation continues, the KV cache grows and slowly overtakes model parameter size. This creates two distinct performance regimes: in the model-dominated regime (small bs), throughput remains relatively stable despite increase sequence length, once we enter the KV-cache-dominated regime, generation speed begins to degrade in proportion to sequence length.

#### Multi GPU inference

#### Batching - the key to good economics
As we've already discussed, batching allows us to spread the cost of loading model parameters across multiple users, decreasing our cost per generated token. This makes us less memory bound, and makes better use of our hardware. At typical sequence lengths, the added KV cache size of an increased batch size is small compared to model parameter loading. **Having sufficient demand and continuously serving big batches is the key to running a profitable LLM inference business; if you can't support large batches, your cost per token will balloon.**

However, this does not scale indefinitely, as bs grows, the kv cache will slowly start to overtake model parameters. When this happens, the cost of loading the model will become increasingly irrelevant to the total time of loading data from the global memory. 

#### Throughput: theory vs practice

