
## Rooflines

Typically computation within a single chip can be overlapped with communication within a chip and between chips. This means that we can lower bound training and inference time by using the maximum of computation and communication time $max(T_{math}, T_{comms})$. The upper bound, when no there is no comp/comms overlap, is therefor naturally given by the sum of $T_{math}$ and $T_{comms}$. We also know that the upper bound and lower bound differ by at most a factor of 2.

If comms and math is perfectly overlapped, and T_math > T_comms, then we are compute-bound. If T_comms > T_math we tend to be communication-bound

The **Arithmetic Intensity** measures of an algorithm measures of many "FLOPs per byte" the algorithm dictates. This is given by the ratio of total FLOPs it performs to the number of bytes it needs to communicate (either intra or inter chips). An accelerator has a peak arithmetic intensity given by its FLOPS divided by its bandwidth. For a TPU v5e this is 240 FLOPs/byte, and for a H100 its 226. This means that if an algorithm has a lower arithmetic intensity than 240/226, it will be bound by byte loading and thus we won't make good use of our hardware. An example of this is a dot product. Take two vectors in bf16 $x, y$ of shape[N]. To perform a dot product we need to load both vectors into memory, each of which has 2N bytes, perform N multiplications followed by N-1 additions and then write back 2 bytes into HBM.

$\text{Intensity(dot product)} = \frac{\text{Total FLOPs}}{\text{Total Bytes}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{N + N - 1}{2N + 2N + 2} \rightarrow \frac{1}{2}$

So the dot product intensity is 1/2, meaning the algorithm performs 0.5 floating point operations per byte loaded, an extremely low value that will be communication bounded on any modern hardware.

**Matrix Multiplication**
A matrix multiplication of two shapes [B, D] x [D, F] will need to load 2BD + 2DF bytes, perform 2BDF FLOPs (think of it as a dot product along the row B and column F between D elements), and write 2BF bytes back. Hence

Intensity(matmul) = 2BDF / 2BD + 2DF + 2BF

If batch size is small relative to D and F, we get that the intensity depends mostly on B. For a bf16 matmul to be compute bound on most TPUs, the local token batch size should be greater than 240.

**Network communication rooflines**
Now, let's instead imagine that we perform a matmul across 2 TPU/GPU. Take the above example again and imagine we split along the D dimension. This matmul is now performed by performing half of each matrix on each accelerator, then copying the partial sums to the other accelerator. 

Intensity (matmul 2 chips) = BDF/2BF = D/2

The algorithmic intensity depends entirely on the dimension D which we've split along. This is because our comms bytes BF stay fixed while math FLOPs scale with D. So the crossover to compute bound occurs when D/2 exceeds the hardwards intensity. 

**Question 1**
1. This is just half of the bytes from before, (2BD + 2DF + 2BF) / 2. 
2. The number of operations are unchanged, its still 2BDF. However, T_math changes because theoretically we should have higher accelerator FLOPs under int8.
3. The intensity is 2BFD / (BD + DF + BF). Making the same assumptions as before we get 
   2BFD/DF = 2B. To calculate the threshold we get 2B = 3.94e14/8.1e11 = 486 <=> B = 243. Which is basically the same as last time. 
4. With the given params we have T_math = 2BFD / 3.94e14 and T_comms = (BD + DF + BF)/8.1e11. Lower bound assumes we have perfect overlap of T_math and T_comms, and then our lower bound is the maximum of the two, and upper bound is T_math + T_comms.

**Question 2**

To determine the batch size we need to first determine the intensity. First, let's figure out the amount of FLOPs necessary. The FLOPs are not influenced by the dtype, hence, we still have 2BFD FLOPs. Now, onto the communication bytes. We load the activations in bf16 and the weight in int8? That would mean load is 2BD + DF. Then we perform the matmul in bf16, and write back the result in bf16 which is 2BF. So that gives us

Intensity(matmul int8 + bf16) = 2BFD / (2BD + DF + 2BF). Under the same conditions as before, we still get 2BFD / DF = 2B. The accelerator intensity is now 1.94e14 / 8.1e11 / 2 = 120. That means if we can do int8 weight quantization, but still do BF16 FLOPs we become compute bound at much lower local token batch sizes, which is great! That is what we want remember.

**Question 4**

Let's look at the FLOPS and bytes. First, we load BD + BDF from HBM, then we perform the matmul which is still 2BDF. Then we store BF to HBM. That gives us an intensity of

2BDF / (BD + BDF + BF) 

Since BDF will dominate the denominator this is essentially = 2. Which means our intensity is constant, which is bad because we will be comms bound no matter what.

**Question 5**
For BFLOAT16 the bf16 FLOPs is 1.979e15 with sparisty and half that without. The HBM is 3.35e12 resulting in a critical batch size of 295.

## TPUs and GPUs

TPUs are fairly simple architectures. They have a high bandwidth memory, HBM which is typically on the order of tens of gigabytes. The HBM transfers data into the TensorCore which performs computation either in the MXU or VPU. Data is transfered through the Vmem. 

![[Screenshot 2025-08-06 at 12.53.37.png]]

The reported HBM bandwidth represents the bandwidth between HBM and the TensorCore (through VMEM). The VMEM is a lot smaller than the HBM, on the order of megabytes, which means data is loaded in chunks from HBM into Vmem and through the tensorcore. This means that the bandwidth between Vmem and MXU is much higher than the HBM, otherwise we would be communication bound a lot. VMEM bandwidth is around 22x higher than HBM bandwidth which means an MXU operation reading from/writing to VMEM requires an arithmetic intensity of only 10-20 to achieve peak FLOPs utilization. That means if we can fit our weights into VMEM instead of HBM, our matrix multiplications can be FLOPs bound at much smaller batch sizes. 

![[Screenshot 2025-08-06 at 12.59.30.png]]
![[Screenshot 2025-08-06 at 13.18.34.png]]

**Question 1** 
Model is 2e11 parameters. Split across 32 TPUv4p. With bf16 that is (400e9 / 32) per chip, and each chip has a HBM bandwidth of 1.23e12. That gives us 10ms. Loading this model into systolic array takes 10ms. This is a lower bound on the latency of sampling from the model, because sampling requires we load all parameters from HBM so it can not take less than 10ms. 

**Question 2**
A v5e pod is 16x16. With a host size being 4x2 that means we have 32 CPU hosts. We have 256 chips and the v5e only had 1 core per chip so that is 256 cores. FLOPS for the pod is 5.0432e16 and the HBM is 4TB.

**Question 3**
So we have to first move A [D, F] and x [B, D] from DRAM into MXU. We know from before that the arithmetic intensity of this matmul is 

2BFD / (2BD + 2DF + 2BF) = (assuming B << D) = B. 

Our T_math is given by 2BFD / 9.2e14. Our communication time is bottlenecked by PCIe so we need (2BD + 2DF + 2BF) / 1.5e10 to transfer data to and from the TPU. Since we want computation to take longer than weight loading, assuming we can completely overlap comms and math, we need 2BFD / 9.2e14 > (2BD + 2DF + 2BF) / 1.5e10. Simplifying this with our assumption of B << D and F = 4D we get 8BD^2 / 9.2e14 > 8D^2 /1.5e10 or B > 61,000

**Question 4**
Weight matrix W int8[16384, 4096], and activation matrix x int8[B, 4096]. On TPU v5e. First, we have to move both from HBM into MUX. We have 67108864 + 4096B bytes to move and a bandwidth of 8.1e11 giving us T_comms = 16384 + B / 49438476.5625 =  . Then, we have to perform the matmul, which is 2 * 16384 * 4096 * B FLOPs, giving us 
T_math = 2 * 16384 * 4096 * B / 3.94e14 = 3.4e-7B.

## Partitioning Notation and Collective Operations

Pop quiz: A's first dimension is partioned across 2x8=16 devices, and the second dimension is not sharded. This means across device meshes X,Y each device holds a int8[8, 2048] array. These are replicated across the Z device mesh. This means each device holds 16384 bytes. In total this uses 524288 bytes in total.

**Case 1 - No Communication Required**
**Lemma:** when multiplying partitioned tensors, the computation is valid and the output follows the sharding of the inputs _unless_ the contracting dimension is sharded or both tensors have a non-contracting dimension sharded along the same axis. For example, this works great 

![[Screenshot 2025-08-08 at 11.19.07.png]]

with no communication whatsoever. Why? Because every device has a local block of A, B that is enough to perform a batch of the computation. This computation is independent of the sharding. Because we have the complete contracting dimension we can perform a complete dot product for a row / col and hold the temporary result. 

**Case 2 - One multiplicand has a sharded contracting dimension**

We need to perform an AllGather. An AllGather removes sharding along an axis and reassembles the shards spread across devices onto each device along that axis.

![[Screenshot 2025-08-08 at 11.56.07.png]]

Then we perform the matmul on each device in full.

When performing an AllGather (or ReduceScatter or AllReduce) in a throughput bound regime. The time, latency of this communication only depends on the size (bytes) of the full array, and the bandwidth (think pcie bandwidth or whatever bandwidth the devices are connected by), NOT THE NUMBER OF DEVICES.

However, there is a caveat. The time for a hop, that is the time it takes to send a shard of the array to two neighbour devices (bidirectional communication), is given by: 

![[Screenshot 2025-08-08 at 12.01.01.png]]

because V is the total size of the shard, X is the number of devices meaning the local shard is V/X bytes, and we need to send it in two directions, so we have bytes/bandwidth. This is all good, but if the ratio of V/X becomes very small, T_hop may be bound by the overhead rather than this formula:
![[Screenshot 2025-08-08 at 12.03.06.png]]

where T_min is the intrinsic overhead of our connection. In this case, the total time does indeed depend on X:
![[Screenshot 2025-08-08 at 12.03.44.png]]

**Pop Quiz**: The per device array is [512, 8192]. The formula says V / W_ICI where V represents the total array size which is [2048, 8192], which should mean it is sufficient to compute the bytes of that array and divide by W_ICI. However, our device mesh is 32, larger than a single host, meaning we have ICI latency. So this gives us 0.00037 seconds. Let's check if we are latency bound, with a mesh of 4 we are doing at most 3 hops which is around 3us which we are long from.

**Case 3 - both multiplicands have sharded contracting dimensions

How expensive is an AllReduce? Well our intuition says that we need to pass the result at each rank to every other device, similar to AllGather. But, we are passing a significantly larger tensor around, because it has the same shape as the full tensor, just with partial results. 

**Question 1** 
The array is sharded across axis X, meaning that across this axis each A is 1/4 the size of A. Along each X axis there are 16 devices. Meaning that each axis along X holds an array of size 16 * A/4 = 4A. That means that total size of A across the device mesh is 4A * 4 = 16A. The ratio is 16. If the array was not sharded it would of taken up a total of 64A.

**Question 2**
We are gathering across axis X, which contains 4 devices. We know from before that T_total = V / W_ICI. However, we have to note here that we are gathering across X but our D is sharded along Y so we only have gather in total 2BD/Y. 

For the AllGather_XY, we now gather across both dimensions XY, meaning that our total bytes gathered is 2BD. We are also working across two axis meaning we have double the bandwidth giving us T_total = 2BD / 2W_ICI = 46us. This is far from the latency bound of 4us (1us per hop) so we are fine.

Finally, an AllReduce. We know that AllReduce=2AllGather. Each shard has size 2BD / (YX) = 2BD/16=BD/8.

**Question 3**
Given that we are latency bound, we can immediately say that the time is given by the latency overhead of one hop (1us) by the number of hops which is 2 so 2us.

**Question 4**
AllGather
Comms: 2DF/W_ICI
Gather across X means sending 2DF bytes.
FLOPS: 2BDF
After communicating we perform a full matmul on X[B,D] * Y[D,F] which is B2DF FLOPs. 

Assuming we can overlap comms/math we get:
T_allgather = max( 2BDF / C, 2DF/W_ICI)

AllReduce
First, we perform the local matmul X[B,Dx] * Y[Dx,F] which gives
T_math = 2BDF/XC
and we end up with Z[B, F]{Ux}. Then we perform an AllReduce which is 2AllGathers, giving us
T_comms = 4BF/W_ICI

T_allreduce = max( 2BDF/CX, 4BF/W_ICI )

The AllReduce is compute bound when D/CX > 2/W_ICI or when D/2X > C/W_ici. Taking v5p as an example we have C/W_ici = 2550. Which means D/5100 > X, so we are compute bound when D is 5100 larger than X which is generally not the case. So with AllReduce we are generally comms bound. Under strategy 1 we are 

comms bound when B < C/W_ICI = 2550 which is often true. So if B < 2550 we are comms bound and we have 

T_comms_allreduce < T_comms_allgather <=> 4BF/W_ICI < 2DF / W_ICI  <=> 2B < D. So if D is twice the size of B then AllReduce is faster than AllGather. That is to say the AllReduce strategy is typically faster if our batch size is small. As the models grow, the AllGather is preferred.   

**Question 5**
This is the case where input is sharded across non contracting dimensions

A[Bx, D]  x B[D, Fyz] -> C[Bx, Fyz]

because this requires no communication, and the matmul flops are reduced by 64 because the each device only performs matmul on its own block 

T_math = 2BDF / 64C


**Question 6**
We are multiplying matrices which are sharded across the contracting dimensions. This is case 3, where we would perform a local matmul giving partial sums 

C[Ix, K]{Uy} 

followed by an AllReduce_Y. Our computation is

T_math = 2IJK/XYC

and then the AllReduce requires the time of 2AllGathers, we are communicating 2IK/X bytes per hop, giving us

T_comms = 2IK/XW_IKI.

**B**

Okay, let's analyze. Both of our output dimensions keep their sharding, and the concatenating dimension is sharded on one of the input matrices. This is the same setting we analyzed earlier in question 4. One option we have is to perform an allgather over J, turning B[Jx, Ky] -> B[J, Ky], allowing us to just perform local matmuls on each rank. To calculat the comms cost, we divide the numer of bytes to communicate which is 2JK/Y

T_comms = 2JK / YW_ICI

Then, performing local matmuls on each rank, where I is sharded over X devices, and K sharded over Y, we get

T_math = 2IJK / XYC

where C is our peak accelerator FLOPs.  

**C**

In the final case we have a simpler variant of the aforementioned question, because this time our contracting dimension J is already replicated, so we don't need to communicate anything. This is Case 1. We can just perform the matmul on each rank:

T_math = 2IJK/XYC

**Question 7**
To reduce latency, we would shard 

B[Dx, F] x D[F, Dy] -> O[Dx, Dy]

## Transformers

x: [P]
y: [P]
A: [N, P]
B: [P, M]

Overall parameter and FLOPs of a Transformer are fairly easy to calculate and are sumarized below using MHA (same num query, key, value heads)

B: batch size
T: sequence length (query)
D: d_model hidden dimension 
F: d_ff hidden mlp dimension
N: Numer of query heads
H: Attention head dimension

A matmul requires 2NPM FLOPs. 
Training FLOPs is 6NPM, because we need 2NPM for the forward pass,
and 4NPM for the backward pass.

| Component | Params per layer | Training FLOPs per layer |
| --------- | ---------------- | ------------------------ |
| MLP       | 3DF              | 18BTFD                   |
| Attention | 4DNH             | 24BTDNH + 12BT^2NH       |
| Other     | D                | BTD                      |
| Vocab     | DV               | 12BTDV                   |
- D = NH (Typically)
- Typically we see F = 4D, which means the parameter count of the MLP block dominates the total parameter count. 
- The total FLOPs budget during training is well approximated as 6 x num_params x num_tokens for reasonable context lengths. (This is not accounting for attention, which starts to dominate at T > 8D)
- During inference, our KV caches are roughly 2SLNH=2TLNH per cache. Meaning it scales linearly with sequence length, number of layers, number of heads, and attention head dimension.

**Question 1**
Total parameters is 17.3B. The attention parameters make up about 25% of the total parameter count. Finally, the KV cache per token is 2TLNH/T = 2LD = 524288, which in int8 is 512KB.

**Question 2**

A[Bx, Dy] x W[Dy, F]. The total FLOPs if nothing was sharded would be 2BDF. Now, since our matrices are sharded each device will perform a matmul of 2BDF/XY. Given the device mesh that means the total FLOPs performed is BDF/16. Since the computation is not sharded across Z we do Z extra flops meaning 2BDFZ total FLOPs. 

**Question 3**

A[I,J,K,L] x B[I,J,M,N,O] -> C[K,L,M,N,O]

Here we assume that I,J are the contracting dimensions. That means we are doing 2KLMNOIJ

**Question 4**

Reshape Q[BTNH] -> Q[BTKGH]
Q[BTKGH] x K[BSKH] -> O[BTSKG]
U = softmax_S(O[BTSKG])
U[BTSKG] x V[BSKH] -> X[BTKGH]

**Question 5**
24BTDNH == 12BT^2NH <=> 2D == T

**Question 7**
The number of training FLOPs divided by the total training seconds gives us the achieved FLOPs / second which is 327 teraflops. The actual accelerator peak fp8 is 1513 teraflops. Giving a hardware utilization of 21.7%.

**Question 8**

We need to load DF weights from memory normally, but E copies in a MoE setup so EDF.

per token FLOPs is 2kBDF. To be compute bound we need an arithmetic intensity over 240, because that is the arithmetic intensity of our accelerator TPUv5e. Arithmetic intensity is given by 

FLOPs / bytes moved

which is 2kBDF / EDF > 240 or kB/E > 120 or B > 120E/k. For DeepSeek this means B > 3840

## Scaling Training

We approximate a Transformer as a stack of MLP blocks, because for large models attention makes up a comparatively small fraction of the FLOPs. As such, we can look at each transformer layer as:
![[Screenshot 2025-08-11 at 14.43.35.png]]
Using our established notation this looks like

In[B, D] x W_in[D, F] x W_out[F, D] -> Out[B, D].

The 4 parallelism schemes can be though of as uniquely defined by a sharding for In, W_in, W_out and Out in the above diagram. Let's go through them.

**Data Parallelism**

In[Bx, D] x W_in[D, F] x W_out[F, D] -> Out[Bx, D].

Naturally, the activations are sharded across the rank X, and communication is not required until the backward pass. Activations, parameters and optimizer states are replicated on each device. 

**Fully Sharded Data Parallelism**
The activations are still sharded across the axis X (just like DP), but now the parameters are also sharded along the same mesh Axis. 

In[Bx, D] x W_in[Dx, F] x W_out[F, Dx] -> Out[Bx, D].

Note that in this setup, due to the parameter sharding W_in[Dx, F] we now a sharded contracting dimension, meaning we require some kind of collective. In FSDP that means we AllGather just-in-time before use in the forward pass. Optimizer states are also sharded.

**Tensor Parallelism**
Activations are sharded across D (d_model), parameters sharded along F. This requires an AllGather and ReduceScatter activations before and after each block. Compatible with FSDP.

In[B, Dy] x W_in[D, Fy] x W_out[Fy, D] -> Out[B, Dy].

Order of operations during forward pass

AllGather_y In[B, Dy] -> In[B, D]
In[B, D] x W_in[D, Fy] -> H[B, Fy]
H[B, Fy] x W_out[Fy, D] -> Out[B, D]{Ux}
ReduceScatter_y Out[B, D]{Ux} -> Out[B, Dy]

**Pipeline Parallelism**
Weights sharded along the layer dimension, activations microbatchd and rolled along the layer dimension. Communication between the pipeline stages is minimal (just moving activations over a single hop). 

#### Data Parallelism
When your model fits on a single chip with even a tiny batch size (>240 tokens, so as to be compute bound), you should always use simple data parallelism. Remember, assuming we can overlap math and comms, when T_math > T_comms we are compute bound. For a matrix multiplication

Intensity(matmul) = 2BDF / (2BD + 2DF + 2BF) ≈ B (for small B relative to D and F)

meaning that when B > Accelerator intensity, which is 240 for a TPUv5e, we are compute bound.

Anyway, like we said, if we can fit the model on a single chip, and have a batch size > 240, we should always use data parallelism. Pure DP splits the activations across the number of accelerators so long as the number of accelerators is smaller than our batch size. The forward pass requires 0 communication, but at the end of every step we need an AllReduce of the gradients in order to sync before updating.
![[Screenshot 2025-08-11 at 15.39.20.png]]

All communication happens **in the backward pass!** There is also a really neat property of the backward pass that the AllReduces are not in the "critical path". Meaning that I don't need dW_out to compute dW_in, and hence we can overlap comms/compute really nicely. The overall communication cost _can still bottleneck us_ if it exceeds our total compute cost, but it is much more forgiving from an implementation standpoint.

**Why do DP?** Pure DP reduces activation memory pressure by splitting our activations over the batch dimension, allowing us to almost arbitrarily increase batch size as long as we have more chips to split over. Especially during training when our activations often dominate our memory usage this is very helpful. 

However, this does nothing to reduce the memory pressure from model params or optimizer states. Typically, with Adam optimizer, we require 10 * num_params memory. Meaning if we have a 96GB TPUv5p pod, the max we can train is 9B model.

**When do we become bottlenecked by comms?** 
We only need to perform comms in the backward pass where we require an AllReduce on a weight matrix W[D, F]. From previous sections we know that an AllReduce is 2x the time of an AllGather, and AllGather comms time is given by the total size of array we are communicating divided by our bandwidth. This is 2 * total bytes / W_ICI. We also require two of these per layer. This gives 

T_comms = V / W_ICI = 2 * 2 * 2 * DF / W_ICI = 8DF/W_ICI

For the matmuls, each layer comprises two matmuls in the forward pass and four matmuls in the backward pass, each of which requires 2(B/X)DF FLOPs. Thus, for a single layer in the backward pass we have:

T_math = 8BDF/XC

Since we overlap

T = max(8DF/W_ICI, 8BDF/XC) = max(1/W_ICI, B/XC)

We become compute bound when B/X > C/W_ICI. This means, for data parallelism to remain compute-bound, we need the per device batch size to be larger than C / W_ICI. This naturally follows from the fact that per device computation time scales linearly with batch size, while communication is independent of this quantity (since we are communicating the weights not the activations). When per device computation is large enough, we are compute bound. Note the striking resemblance of the B > C/W_ICI condition to the single device compute-bound rule B > 240; in that case as well the rule came from the fact that computation time scaled with batch size while data-transfer size was independent of batch size. 

Putting some real numbers on this thing, for TPUv5p we need our batch size to be at least 2550 to avoid being communication bound. Since we can DP over multiple axes, if we dedicate all three axes of a pod to DP, we 3x our bandwidth and can scale down to only BS=850! Remember, we use B to refer to the **total batch size in tokens**. Clearly, however, a batch is made of K sequences of T tokens each, so how can we do this? As far as the MLP goes, this does not matter, tokens are tokens, they are processed independently. You are free to do parallelism over both batch and sequence dimension: this is called context or sequence parallelism, but it is data parallelism. 

#### FSDP
FSDP is a variant of data parallelism that splits the model parameters and optimizer states across the data parallel shards and efficiently gathers and scatters them as needed. Compared to pure DP, FSDP drastically reduces per device memory usage and saves on backward pass FLOPs, with very minimal overhead.

![[Screenshot 2025-08-11 at 20.38.43.png]]

Syntax: In[Bx, D] x W_in[Dx, F] x W_out[F, Dx] -> Out[Bx, D].

Remember that an AllReduce can be decomposed into an AllGather and a ReduceScatter. FSDP applies this to DP. Instead of doing full gradient AllReduce, FSDP shards the weights and optimizer states across chips, AllGather them at each layer during the forward pass and ReduceScatter across the weights during the backward pass at no extra cost.  FSDP has basically no overhead compared to DP. 

Standard DP involves a lot of duplicated work. Each TU AllReduces the full gradient, then updates the full optimizer state (identical work on all TPUs), updates params (again duplicated). For ZeRO sharding (sharding gradients/optimizer state), instead of the AllReduce, we ReduceScatter the gradients, update only your shard of the optimizer state, update a shard of the parameters. 

FSDP has the same relative FLOPs and comms cost as pure DP

T = max(B/XC, 1/W_ICI)

This is great because it means if our per device batch size is big enough to be compute-bound for pure DP, we can - without worrying about leaving the compute-bound regime - simply upgrade to FSDP, saving a massive amount of parameter and optimizer state memory. Although we do add comms during the forward pass, the cost is immaterial since it overlaps with forward pass FLOPs.

As a concrete example, DeepSeek-V2 used a batch size of 40M tokens. That would allow you to scale to 47,000 chips before hitting the bandwidth limit. 

	Takeaway: FSDP and pure DP become bandwidth bound on a TPUv5 when B < 2550/n_axes

#### Tensor Parallelism
In a FSDP AllGather we move the weights across chips. We can also shard the feedforward dimension of the model and move the activations during the layer - this is called "1D model parallelism" or megatron sharding. This can unlock a smaller efficient batch size per pod.  

Syntax: In[B, Dy] x W_in[D, Fy] x W_out[Fy, D] -> Out[B, Dy] 

**Gather Weights vs Activations**
In FSDP, looking at the flow, we have an input which is sharded along the batch dimension and the weights are sharded across D. 
 In[Bx, D] x W_in[Dx, F] -> Tmp[Bx, F]

This means, to be able to perform a matmul, we will have to perform an AllGather_X on W_in
AllGather_X W_in[Dx, F] -> W_in[D, F]
before we can compute the matmul. This means we gather the **weights** across our network.

In TP, our syntax is
In[B, Dy] x W_in[D, Fy] -> Tmp[B, Fy]
Which means we now have to gather the **activations** across Y to be able to perform the matmul.

This is cheaper than ZeRO sharding when the activations are smaller than the weights! This is typically true only with some amount of ZeRO sharding added (which reduces the size of the gather). This is one of the reasons we tend to mix ZeRO sharding and model parallelism. TP comms cost is BD, and FSDP is DF. Meaning TP is cheaper than FSDP when

BD < DF    <=>     B < F

the batch dimension is smaller than the feed forward.

**TP Algorithm**
In[B, D] = In[B, Dy] AllGather
In[B, D] x W_in[D, Fy] = Tmp[B, Fy]
Tmp[B, Fy] x W_out[Fy, D] =  Out[B, D]{Uy}
Out[B, Dy] = ReduceScatter(Out[B, D]{Uy})

One nice thing about TP is that it interacts nicely with our matrices in the Transformer forward pass, because we can do a AllGather to start, and only have to do a ReduceScatter Out at the end. What is the cost of this (modelling only the forward pass)

T_comms = (2BD + 2BD) / W_ICI = 4BD/W_ICI
T_math = 2BDF/YC + 2BFD/YC = 4BDF/YC
T = max(4BDF/YC, 4BD/W_ICI) = max(F/YC, 1/W_ICI)

Noting that we want compute cost to be greater than comms cost we get

F/YC > 1/W_ICI       <->.       F > YC/W_ICI

On a TPUv5p we have C/W_ICI 2500 in bf16 meaning we are compute bound when F/2500 > Y. So, for a given feed forward dimension F, we can only increase our mesh axis Y up to a certain point before we become comms bound. 

	Takeaway: Model Parallelism becomes comms bound when Y > n_axes * F/2500. For most models this is between 8 and 16 way model parallelism.

- On a TPU4vp with Llama 3-70B that has D=8192, F=30,000 we can comfortable do 8-way model parallelism but will be communication bound on 16 way model parallelism. 
- For Gemma 7B we become comms bound at 19-way MP.

#### Mixed FSDP and TP

Syntax: In[Bx, Dy] x W_in[Dx, Fy] x W_out[Fy, Dx] -> Out[Bx, Dy]

FSDP and TP are easily combined, by sharding W_in and W_out along both axes we both save memory and compute. Because we shard B along X, we reduce the size of the model-parallel AllGathers and because we shard F along Y, we reduce the communication overhead of FSDP. 

**Algorithm**
In[Bx, D] = AllGather_Y(In[Bx, Dy])
W_in[Dx, Fy] = AllGather_X(W_in[Dx, Fy])
In[Bx, D] x W_in[D, Fy] = Tmp[Bx, Fy]
W_out[Fy, D] = AllGather_X(W_out[Fy, Dx])
Tmp[Bx, Fy] x W_out[Fy, D] = Out[Bx, D]{Uy}
Out[Bx, Dy] = ReduceScatter(Out[Bx, D]{Uy})

There are more communications, but note that they are smaller. For example, the first step of the TP algorithm was to AllGather(In[B, Dy]), this collective is now reduced by a factor X because we only move AllGather(In[Bx, Dy]).

A simple but key maxim is that FSDP moves weights and TP/MP moves activations. That means as our batch size (especially as we do more data parallel, meaning higher X), model parallelism becomes cheaper, as we noted above, because the per shard activations In[Bx, D] that we move are smaller. 

- Model parallelism performs AllGather_Y([Bx, Dy]), which shrinks as X grows
- FSDP performs AllGather_X([Dx, Fy]) which shrinks as Y grows

Thus by combining both we can push our minimum batch size replica down even more. Let X be the number of chips used to FSDP, and Y the number of chips dedicated to TP. Let N be the total number of chips N=XY. Mx is the number of mesh axes over which we do FSDP and My for TP.

Total bytes communicated by each collective:
AllGather_Y(In[Bx, Dy]) = BD/X
AllGather_X(W_in[Dx, Fy]) = DF/Y
AllGather_X(W_out[Fy, Dx]) = FD/Y
ReduceScatter(Out[Bx, D]{Uy}) = BD/X

T_FSDPcomms = 4DF/YW_ICI
T_TPcomms = 4BD/XW_ICI

Our total flops is 

T_math = 2BDF/XC + 2BFD/YC = 4BDF/NC

Under the assumption that we do not overlap comms on the X and Y axis, the total comms time is

T_comms = T_FSDPcomms + T_TPcomms

First, let's identify the optimal values for X and Y to minimize total communication. Since FLOPs are independent of X and Y, the optimal settings are those that simply minimize comms. We find that:
![[Screenshot 2025-08-13 at 10.16.39.png]]

meaning that for a given B,F,N we know what amount of FSDP is optimal. Plugging in real values, N=64 (corresponds to a 4x4x4 array of chips), B=48000, F=32768 gives X ≈ 13.9. So we would choose X to be 16 and Y to be 4. 

	Takeaway: Combining TP and FSDP allows us to drop to a per device batch size B/N of 2*2550^2/F. 
	
![[Screenshot 2025-08-13 at 10.29.43.png]]

**Summary**

- If we can fit the whole model in memory, and our batch size is > 240, we should always do pure DP.
- For a TPUv5p C/W_ICI ≈ 2,550.
- Pure data parallelism is compute-bound when B/X > C/W_ICI, meaning when the per-device batch size is large enough.
- FSDP is compute-bound when B/X > C/W_ICI, same as for DP. Meaning that if our per device batch size B/X is large enough, we can simply upgrade to FSDP and save a lot of memory.
- Tensor Parallelism is compute bound when F > Y2500. This is independent of batch size, and is around (Y=) 8-16 way for most models.
- Mixed FSDP + TP allows us to drop the batch size to 2 * 2550^2/F ≈ 400.

If we have LlaMA-3 70B with F≈30,000 on TPUv5p. We will be comms bound under model parallelism at Y > n_axes * F/2550 ≈ n_axes * 11. So anything over 11 way per axis.

Pure FSDP becomes ICI bound when the per device batch size is < 2550/n_axes. That means, if we want to use a total batch size of 2M, the most amount of chips we can use is:

B/X < 2550/n_axes <-> 2e6/X < 2550/3 <-> X = 2400

Mixed FSDP + TP means you are ICI bound at batch size < 432. That means, with 2 axes, we get 

2e6/X < 432/2 <-> X = 9259 chips.

**Takeaways**

- Generally, increasing parallelism, or reducing the batch size, both tend to make us more communication bound, because they decrease the amount of per device computation.
- Up to a reasonable context length (~32k) we can get away with modeling a transformer as a stack of MLP blocks and define each of several parallelism schemes by how they shard the two/three main matmuls per layer.
- During training there are 4 main parallelism schemes we consider, each of which has its own bandwidth and compute requirements.

| Strategy                                  | Description                                                                                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data Parallelism                          | Activations are batch sharded, everything else is fully replicated, we all reduce gradients during the backward pass                                 |
| FSDP                                      | Activations, weights and optimizer are sharded, weights are gathered just before use, gradients are reduce scattered.                                |
| Model Parallelism (aka megatron / tensor) | Activations are sharded along d_model, weights are sharded along d_ff, activations are gathered before W_in, the result reduce scattered after W_out |
| Mixed FSDP + Model Parallelism            | Both of the above                                                                                                                                    |
|                                           |                                                                                                                                                      |
|                                           |                                                                                                                                                      |
Syntax:

![[Screenshot 2025-08-13 at 13.11.06.png]]

**Question 1**
The MLP layers has 3 matrices, each of which with DF parameters
FFN: 3DFL

The attention is 4 up projections W_QKVO matrices with shapes DNH and DKH, because H=N we get
Attention: 4DNHL

Vocabulary params = 2VD

**Question 2**
BS=16M tokens, Adam optimizer.

Parameters in bf16: 13B * 2 = 26GB

Optimizer state: First moment estimate and second moment estimate per parameter stored in fp32. That is 8 bytes per param: 104GB

The activations after each matmul are shape BF, BF, BD. At bf16, these take up: 4BFL + 2BDL =  2LB(2F + D) = 4.19e13 = 42TB

**Question 3**
Some numbers first. 32k sequence length and a 3M batch size gives us a sequence batch size of 96. On a TPU v5p 16x16x16 we have 393TB of HBM

1. Using pure data parallelism, the model is replicated across the devices, and because the param + optimizer states take up 130GB we need that to fit on device. TPUv5p have 96HBM, so they dont fit.
2. Pure FSDP means we shard parameters, optimizer states and activations across the batch. Replacing 16M with 3M in Question 2, we get activations of size 7.86e12. We also have 1.3e11 in optimizer states, bringing us to almost exactly 8e12 = 8TB. This means we are well under the limit of 393TB of HMB. Remember the activations and optimizer states are sharded across our slice. The per device memory usage is 1.95GB. Let's not analyze if we are compute or communication bound. We know from earlier that under FSDP we are comms bound if B/X < C/W_ICI * n_axes. For a TPUv5p with 3 axes our arithmetic intensity is 850 -> B/X < 850. Our per batch size is 3e6/16^3 = 732. So yes we are comms bound. Or, another way of looking at it is that with 4096 chips and a minimum intensity of 850 we need a minimum batch size of 3.48M.
3. Mixed FSDP + TP. This means we are comms bound if B/X < 2 * 2550^2  / F = 940. That means its actually worse than FSDP. 

**Question 4**
Dropping the batch size to 1M means we have a 244 per device batch size. That is barely enough to be above our on device computation bound threshold. 

### Training Llama 3 on TPUs
A practical example of training the LLama 3 family herd of models. The herd includes 8B, 70B and 405B models.

Starting off with the 70B model, let's have a closer look at it:

![[Screenshot 2025-08-13 at 14.48.06.png]]
##### Params
Per layer params
FFN: 3DF
Attention: W_Q and W_O have DNH, W_K and W_V have DKH. In total we have 2DNH + 2DKH.
Vocab: 2VD

In total that gives L(3DF + 2DNH + 2DKH) + 2VD = 70.52B

The FFW make up 80% of the total parameter count, Attention is 17% and output embeddings are 3%.

##### FLOPS
The flops are typically 6 * num_params, meaning 420B FLOPs per token. That means we are doing roughly half a TFLOP per token per training step. Assuming we are compute-bound, this should take 0.9ms on a TPUv5p.

To calculate exactly: The FFN layer has 3 big matmuls, each of which takes 2BTDF FLOPs. That is in total 3 * 2BTDF * 3 = 18BTDF, where the second factor 3 comes form the fact that we do 1 of these in the forward pass and 2 for the backward pass. The attention consists of 6 large matmuls, and the dot product attention. Ignoring the dot product flops, and just taking the 6 matmuls, they require

2BTDNH + 2BSDKH + 2BSDKH + 2BTKGS + 2BTKGH + 2BTNHD = 4BTNHD + 4BSDKH + 2BTKGS + 2BTKGH. 

Given 15T training tokens, that gives us a total FLOPs of 6.3e24. If we are compute-bound this would take 158k days on a single TPU v5, roughly 435 years. 

If we assume a full TPU v5p pod with 8960 chips and a MFU of 40% this would take around 1000 hours  ≈ 44 days. Which is fairly reasonable, assuming we can achieve 40% MFU. 

Llama 3 70B was pretrained with a batch size of 4M tokens. The optimizer state will take up
70 * (2 + 4 + 4) = 700GB. Then for the activations, we are checkpointing 4 times per layer, it depends on which shape we are checkpointing, but let's assume they are all B,D matrices, that means

2 * 4e6 * 8192 * 4 * 80 ≈ 21TB

That means we need atleast 21.6TB of HMB. Since each TPU has 96GB of HBM we need 225 TPUs at least. But, as we've already established, the total FLOPs to train is 6.3e24, which if our 225 TPUs were running with 100% MFU, would take 706 days to complete. It becomes quite evident that the reason our clusters are growing isn't due to memory, but rather FLOPs. 

If we assume we are using 8960 chips, that means we have about 2.4GB per chip which is basically nothing.

#####  How to shard for training
Assuming the same setting from before, we want to train LLaMa 3 70B with a 4M token batch size (1024 sequences of length 4096 per batch) on a TPU v5p pod of 8960 chips. Let's try to identify the best sharding strategy.

**Question:** Under the assumptions above, can we train our model with FSDP alone? To start, let’s say we can’t do any sequence/context parallelism. _This should be the first idea you have, since it’s simple and will introduce no extra communication if it works_

FSDP shards parameters, activations and optimizer states, meaning that memory wise, we will fit on 8960 no problem. The question however, says that we can't do sequence/context parallelism, that means that because we only have 1024 sequences, we can at most split across 1024 chips.

**Question:** Let’s relax the requirement of not doing any sequence sharding. If we allow ourselves to do FSDP over both the batch _and_ sequence axes, can we train LLaMA 3-70B with only FSDP on 8960 chips?

Okay, now this is a normal FSDP setting as discussed before. Remember, FSDP is communication bound when B/X < C/(W_ICI * n_axes). We're using TPU v5p so C/W_ICI is 2550, and in this case n_axes is 3 i think. So we are comms bound when the per device batch size is < 850. With the given setup, our per device batch size is 4e6 / 8960 = 446. So we are comms bound.

**Question:** Now let’s look at mixed tensor parallelism and FSDP. Does there exist some combination that lets us remain compute-bound? What amount of FSDP and tensor parallelism should we do if so?

Let's see, for mixed FSDP + TP, we have established that our condition for comms bound is 
B/N < 2550^2 / (MxMyF) = {where MxMy must be 2 in a 3D mesh} = 2550^2/2F = 113. That means we can be compute bound! To compute the optimal amount of FSDP/TP we go to our trusty derivation

X_opt = sqrt(B * Mx * N / F My) ≈ 1618. If we round to the nearest multiple of 2 we get 2048. That means we should do 2048 

### Inference

Naive sampling from a transformer. Put prompt in, get log p(next token | previous tokens). Sample from distribution, put prompt + next token in. Repeat.

![[Pasted image 20250817111229.png]]

This works, but we never do this in practice. Due to the causal dependency of the transformer decoder, token $n_t$ only depends on $n_{t-1}$, so, at the second step in the image above we are recomputing the same thing for all previous tokens that we already processed in step 1. The forward pass is (n²) on the FFW and O(n³) on the attention mechanism to generate n tokens, that is expensive!!
  

Instead of doing the full forward pass every time, we can save some intermediate activations from each forward pass that allows us to avoid re-processing previous tokens. Specifically, since a given token only attends to the previous tokens during dot product attention, we can simply write each token's key and value projections into a new data structure called the **kv cache**. Once we've saved these key/value projections from past tokens, future tokens can simply compute their $q_i \cdot k_j$ products without performing any new FLOPs on earlier tokens. Amazing! This naturally divides inference into two separate stages

**Prefill**: This is the first step in the image above, where we have yet to process the prompt. At this step we process **all** the tokens in the prompt at the same time, saving resulting activations (specifically key-value projections) in a KV cache. We also save the logits for the last token.

**Generation**: Given a KV cache and the previous logit, we sample a new token and feed that token into the Transformer and produce a new set of logits. We also append the new KV activations to the KV cache.

Here's a new visualization with a KV cache

![[Pasted image 20250817111307.png]]
  
By sampling with a KV cache we reduced our time complexity to generate n tokens to O(n) in the FFW and O(n²) on the attention, since we never reprocess a previous token. We will see that prefill and generate are two **very** different tasks, with the KV cache being a novel and significant source of complexity.


**What do we want to optimize?**
A part of inference that's totally new compared to training: *latency*. During training we focus on throughput, the total tokens processed per seconds, during inference we have to worry about how fast we're producing tokens, measured as both **Time to First Token** (TTFT) and the **per token latency**. This is different for different use cases:

- Chat interfaces / streaming tasks need to run cheaply at while while having a low TTFT, generating tokens fast enough to exceed human speed

- Offline batch inference for evals and data generation only care about the bulk cost of inference and is blind to the latency of individual samples

- Edge inference only needs to service one user at a time at the lowest possible latency.

Maximizing hardware utilization is still critical and helps with cost and TTFT, but unlike training it does not *necessarily* translate to better experience for individual users in all contexts. Many optimizations at the accelerator, systems and model arch level make tradeoffs between latency, throughput, context length and model quality.
#### A granular view of the Transformer

Before, when we were looking at the training perspective, we treated Transformers as a stack of MLP layers. While this is often reasonable from a FLOPs and memory standpoint, it is not sufficient to properly model inference. The major components of the Transformer forward pass are:

1. **a bunch of linear operations**, including the MLP: W_in and W_out; the attention QKVO projections: W_Q, W_K, W_V, W_O. These all involve reading parameters and a batch of activations from HBM, doing some flops and then writing the result back to HBM.

2. **dot product attention** We need to read a batch of key-value projections and a batch of query activations from HBM, do a few inner products and some softmax operations and write back to HBM.

3. **everything else** including layer norms, activation functions, token sampling, updating kv cache and pos embeddings. These take some FLOPs but are dominated by, or fused into the above
#### Linear operations: what bottlenecks us?

Let's look at one of the linear operations, which take the form of a bf16[B, D] batch by a bf16[D, F] weight matrix. This could be either one of the big W_in/out in the MLP block or one of the smaller attention projections. To perform this matmul we need to load into HBM, perform the matmul, and store back into HBM. That means we have to move 2BD + 2DF weights into HBM, perform matmul, and then store back 2BF. Let's assume a TPU v5e, the time this takes is given by

T_comms = bytes moved / bandwidth = 2(BD+DF+BF) / W_HBM  

Then the matmul we are performing is obviously 2BDF FLOPs, and the time it takes it

T_math = computation FLOPs / accelerator FLOPs = 2BDF / C

We are compute bound if

T_math > T_comms = computation FLOPs / accelerator FLOPs > bytes moved / bandwidth = computation FLOPs / bytes moved > accelerator FLOPs / bandwidth = intensity(algorithm) > intensity(TPU v5e)

where intensity(TPU v5e BF16) = 1.97e14 / 8.1e11 = 243

With this we get that  

2BDF / 2(BD+DF+BF)> 243

which can have different characteristics depending on the size relationbetween B,D,F. Typically F>D>>B which gives  

BDF / DF(B/F + 1 + B/D) <-> B -> B > 243 = B_crit

If we quantize our weights or use lower precision FLOPs for the matrix multiplication this critical batch size can change. For instance if our weights are quantized in int8 the bytes we get

2BDF / (2BD + DF + 2BF) <-> 2BDF / DF(2B/F + 1 + 2B/D) <-> 2B -> B_crit = 243/2

or if we do our FLOPs int8 / fp8 we now load everything in int8 meaning

2BDF / BD+DF+BF <-> 2B -> 2B > HMB int8 intensity = 3.94e14 -> B > 243

so basically nothing changes if we do things in int8. We are moving 2x less data which reduces communication load, but our accelerator is 2x faster so it evens out.  

We can draw some general conclusions from this; if we let $\beta$ = bits per param / bits per activation, and alpha_hbm = intensity(accelerator) = C/W_hbm, then our critical batch size is
B_crit = $\beta*\alpha$.

	Takeaway: Transforme matmuls are compute bound iff the per replica token batch size is greater than B_crit = C/W_hbm * (bits per param / bits per activation) = beta*alpha. For bf16 activationson a TPU v5e this is 240 tokens, for an h100 this is about 280 tokens.

Remember that batch size here refers to the token batch size. During training, we'll have a very high algorithmic intensity because we reuse the same weights over a very large batch. This high intensity carries over to prefill since user prompts are typically hundreds if not thousands of tokens long. If a sequence is longer than 240 tokens and fed into a dense model we expect it to be compute-bound and all is well. Prompts shorter than this can technically be batched together to achieve higher utilization but this is typically not necessary.

	Takeaway: During prefill, all matrix multiplications are basically always compute-bound. Therefore simply maximizing hardward utilization or MFU is enough to maximize throughput per chip (cost) and latency (in the form of TTFT). Unless prompts are extremely short, batching at a per-prompt level only adds latency for a small improvements in prefill throughput.

However, when we move to the decoding/generation stage we can only do our forward passes one token at a time. Thus we can only (easily) achieve good utilization by batching multiple requests together, parallelizing over the batch dimension. Apparently, batching over concurrent requests is hard without affecting latency, for that reason it is much harder to saturate the hardware FLOPs with generation.
  
	Takeaway: During generation, the total token batch size must be greater than B_crit to be compute bound on the linear/feed forward operations. Because generation is only done on one token this requires batching multiple requests, which is hard

You have to realize that handling **240 concurrent requests** means handling 240 separate KV caches. That means this is difficult to achieve in practice. In contrast, pushing more than 240 tokens through during the prefill is pretty routine.
#### Attention!

Things get more complicated as we turn to Attention :) Looking at pure multi head scaled dot product attention. In a single Flash Attention fusion we, ignoring
softmax, masks etc, we:

- Read Q activations of shape bf16[B, T, D] (assuming D=NH) from HBM
- Read the KV cache which is a pair of bf16[B, S, D] tensors from HBM
- Perform 2BTSD FLOPs in the QK matmul, with flash attention we dont need to write bf16[B,S,T] attention matrix mack into HBM
- Perform AV matmul taking 2BTSD FLOPs
- Write the resulting bf16[B,T,D] tensor back into HBM.

Putting this together we get

Multihead attention intensity = FLOPs / bytes moved = 4BTSD / 2BTD + 2BSD + 2BTD = TS / T + S

During prefill S=T giving us T/2. This is great becuse it means the arithmetic intensity of attention during prefill is O(T). That means it is quite easy to be compute-bound in attention, as long as our sequence length is fairly large. But, during generation S>>T = 1 giving us

ST/(T+S) = S/(S+1) = 1 as S grows.

This is bad, since we cannot do anything to improve the arithmetic intensity of attention during generation. We're doing a tiny amount of FLOPs while loading a massive KV cache. So we are basically always memory bandwidth bound during attention.

	Takeaway: During prefill, attention is typically comput bound for any reasonable sequence length (roughly > 480 on a v5e), while during generation our arithmetic intensity is roughly 1 and constant, so we are ALWAYS memory bandwidth bound.

Let's think about this. During the linear portions of the model we are compute bound because the parameters (the memory bandwidth heavy components) are reused over many batch items. However, every batch item has its own KV cache, so a bigger batch size means more kv caches. We will almost always be memory bound here unless the architecture is adjusted.

#### Theoretical estimates for LLM latency and throughput
*The following are among the most important to take away from the inference chapter*

We are considering generation here, we established above that attention has a constant intensity during generation ≈ 1. Consider what happens when we increase our batch size during generation.

For small batch size during generation (which is common) we can lower bound our per step latency by assuming we are memory bandwidth bound in attention and MLP. Naturally, the time is given by bytes moved / bandwidth. In this case, we are looking at the total forward pass time, that means we have to load **all parameters** once, as well as load the kv cache for every element in the batch. Hence we get:

Theoretical Min Step Time = Batch Size x KV Cache size  + Parameter Size / Total Memory Bandwidth

We can visualise this relationship. As discussed earlier, at small batch sizes the parameter loading dominates, but as the batch size grows the kv cache size slowly edges closer and eventually dominates the parameter loading.

![[Screenshot 2025-08-17 at 11.55.55.png]]

Now, if we instead imagine the throughput. Throughput is the amount of tokens produced per unit time. During generation, each step produces exactly Batch Size new tokens and each step takes min step time (as above). That means we divide batch size by the step time and we get throughput:

Theoretical Max Token/s = Batch size x Total memory band / Batch size x kv cache size + parameter size

Which, as the batch size grows, asymptotically approaches Memory bandwidth / kv cache size per item. 

You get diminishing returns from increasing batch size. This is because as batch size grows, FLOPs begin to dominate, we have to look at the more general time equation to see this. Attention is always memory bound, but MLP depends on the batch size:

$$\begin{align} \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\text{MLP (can be compute-bound)}} \end{align}$$

**Pop Quiz** Assuming we are memory bound in both attention and MLP, we get 

min_step_time = 4 * 8192 * 100e3 + 30e9 / (16 * 8.1e11) = 2.5ms

At 256 tokens we may be into the compute bound regime, we need to check:

T_comms_mlp = 30e9 / 16 * 8.1e11 = 2.3ms
T_math_mlp = 2 * 256 * 30e9 / (16 * 1.97e14) = 4.87ms

so we are well into the compute bound regime for our MLP. That means our step time is

T = 256 * 8192 * 100e3 / (16 * 8.1e11) + 2 * 256 * 30e9 / (16 * 1.97e14) = 21ms

As we increase our batch size, the step time increases proportionally, providing a clear tradeoff between latency and throughput. And as the MLP becomes compute bound, step time increases at a faster rate, albeit with better efficiency. Small bathes are fast but dont utilize hardware well. Bit batches are slow but efficient. 

#### Design an effective inference engine

TOo avoid wasted TTFT from batched prefill, while keeping generation throughput high, we can implement a simple improvement called interleaved. This is where we prefill with batch size 1, and generate in batches. A simple toy comparison

Given 4 arriving requests:
- **Naive batching**: Process all 4 prefills together (high TTFT due to padding/waiting), then generate for all 4
- **Interleaved**:
    1. Prefill request 1 → start generating for request 1
    2. Prefill request 2 → now generate for requests 1+2 together
    3. Prefill request 3 → generate for requests 1+2+3
    4. Prefill request 4 → generate for all 4
    5. Continue generation steps for all active requests until they complete

In the interleaved configuration, the default state is generation, and as soon as we get a new request (and we've completed a generation step), we switch over to prefill mode and perform the prefill, then add that do our generation batch and keep generating.

The main disadvantage of this is that generation requests are paused as soon as a new prefill is performed, which means that other users prefills are on the critical path of a users generation. The token generation will therefore be jittery, and still quite slow on average, even though the TTFT is improved. 

The next natural step is to completely separate decode and prefill. This means running the two tasks completely separately, even on two sets of TPU/GPU if possible. The prefill server generates KV caches that get sent across the network to the generate servers, which batch multiple caches together and generate tokens for each of them. This is called **disaggregated** serving. 

![[Screenshot 2025-08-20 at 09.45.44.png]]

This provides

1. Low latency at scale: A user's request never blocks on another users, except if there is insufficient prefill capacity. The request is immediately prefilled, then sent to the geeneration server, then immediately slotted into the generation buffer. 
2. Specialization: Quite often the latency optimal parameter sharding strategy for prefill and generate is different. Constraining the two operations to use the same sharding hurts the performance of both, and having two sets of weights uses memory. Also, separating the two means the prefill server doesn't have to hold any KV caches except the one its currently processing. 

The downside is of course that the KV cache has to be transfered across ICI. This is typically acceptable but again motivates for reducing KV cache size.

	Takeaway: for latency sensative, high throughput serving, we typically have to separate prefill and generation into separate servers with prefill operating at batch 1 and generation batching many concurrent requests together.


#### Worked Problems

For the following problems we assume a made of LLama 2 13 model with the following configuration

| hyperparam       | value  |
| ---------------- | ------ |
| n_layers (L)     | 64     |
| d_model (D)      | 4,096  |
| d_ff (F)         | 16,384 |
| n_heads (N)      | 32     |
| n_kv_heads (K)   | 8      |
| d_qkv (H)        | 256    |
| n_embeddings (V) | 32,128 |

**Question 1**. Again we've done this a lot of times, the size is L(FFW + Attn) + Vocab = (L*(3*D*F + 2*N*H + 2*K*H) + D * V) / 1e9 = 18.4B. The KV cache per token is 2 * bytes * L * K * H = 0.5MB.

**Question 2**. 
For a TPUv5e 4x4 we have 16 * 16 = 256GB HBM. Our parameters take up 18GB with int8. That leaves 238GB. We know that for inference, we have to store parameters, and the KV cache, and our activations are negligible. The KV cache uses 256KB per token in int8, that means we can fit 930K tokens in total. 

Dropping our KV heads to 1 would reduce our per token kv cache to ~32KB. That means we can fit 7.4M tokens. The amount of tokens we can fit scale proportionally to our KV heads. 

**Question 3**
Right, when we perform inference, much time is spent loading parameters into MXU. If this is what we are bound by, it provides a good lower bound on our step time. To calculate this time we divide the bytes moved by our bandwidth speed. Let's assume our mesh dimensions are X, Y, and that we shard across both.

T = (18e9 / (XY)) / W_HBM = 18e9 / (8.1e11 * 16) = 1.38ms

**Question 4**
We are typically memory bound in this scenario, that means our the lower bound for each individual device is the time it takes to load the necessary parameters. When we perform tensor parallelism I think we want to make sure that the time spent ICI is less than HBM data transfer. Essentially we want to remain memory bound as opposed to communication bound.

The upper bound on TP over ICI is given by the fact that we have to gather activations

T_ICI = B * D / W_ICI

our HBM time is 

T_HBM = (DF / XY) / W_HBM

which gives

T_ICI > T_HBM <-> BD/W_ICI > DF/(XYW_HBM) <-> BXY/F > W_ICI/W_HBM 

plugging in our parameters for TPU v5e we have W_ICI = 9e10, W_HBM=8.1e11, and XY=16, F=16384

B > 113.

BY/F > W_ICI/W_HBM <-> Y > F/B * W_ICI/W_HBM <-> Y > 1820/B

That means as our batch size grows, the amount of TP we can do shrinks. For example:
- If B = 1, then Y > 1820 (can do a lot of TP)
- If B = 100, then Y > 18.2 (limited to ~18-way TP)
- If B = 240, then Y > 7.6 (limited to ~8-way TP)

We also have to shard our KV cache. Our model has K=8, which is less than our available 16 TPUs. But we only have 2 mesh dimensions available, and we are already using 1 for TP so we only have 1 dimension available. That means we will stick to just sharding along the head dim as KV[2, B, L, S, Ky, H].

Under this sharding strategy, assuming we are HBM bound and not ICI bound, our per step latency is given by the time to load parameters and kv cache for attention. The KV cache is sharded across X, and the params are sharded across Y. This gives

step time = B * KV cache size / YW_HBM + params / XW_HBM = B5.5e3

meaning at B=1 we have a latency of 5.5ms

## Serving LLaMA

Looking at the 70B version again

|**hyperparam**|**value**|
|---|---|
|nlayers (L)|80|
|dmodel (D)|8,192|
|dff (F)|28,672|
|nheads (N)|64|
|nkv heads (K)|8|
|dqkv (H)|128|
|nembeddings (V)|128,256|
This model has a KV cache of [2, L, K, H] per token, assuming int8, that is 163KB. That means just a single sequence of 8192 length requires 1.34GB of memory. 

Serving this model with a batch size 32 and 8192 sequence length in int8 would require:

Params: 70GB
KV Cache: 2 * L * K * H * 32 * 8192 = 43GB

That is a total of 113GB, which would require a 4x2 TPU v5e, or perhaps even a 4x4 to account for overhead.

To calculate the **decode latency** we look at 

$$\begin{align} \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\text{MLP (can be compute-bound)}} \end{align}$$

First, calculating attention time is straight forward

T_attn = B x kv cache / 8 * W_HBM = 32 * 8192 * 160e3 / (8 * 8.1e11) = 6.47ms

Remember that this is decoding / generation, so T = 1 and S=8192. For the MLP to be compute bound we need a per device batch size > 120 (params int8). In our config, the per device batch size is 32/8=4, so we are well into the memory bound regime. That gives:

T_mlp_comms = 70e9 / 8 * W_HBM = 10ms

Meaning our per step latency is 16.47ms and our latency is 32 / 17e-3 = 1882 tokens/second  or 235 tokens / sec / chip.

The **one caveat to check here is if we are ICI bound** on our matmuls. In the above equations we have assumed that we are HBM bound, so we need to make sure this is true. In theory we are ICI bound if Y > n_axis * F/2200 = 26. Let's remind ourselves where this comes from.

TP:
In[B, Dy] * D Win[D, Fy] * Wout[Fy, D] -> Out[B, Dy]

Where we AllGather the activations before the first matmul, then ReduceScatter them after the second. There are two matmuls, requiring 2BDF/Y + 2BFD/Y FLOPs

T_comms = 2 * 2 * B * D / W_ICI
T_math = 4BDF / YC

T_comms < T_math <-> 4BD/W_ICI < 4BDF/YC <-> YC/W_ICI < F <-> {C/W_ICI = 2200 on a TPUv5e} <-> Y < F/2200.

That means we are compute bound as long as Y < F/2200. Note that this is independant of the precision of the computation. As an example, under int8 flops C_int8/W_ICI will double, but at the same time our communication volume is halved, so the two factors cancel. 

If we were to run on a 4x4 we would still be fine ICI-wise and our latency would drop to by the same factor as we increase our number of TPUs, so down to 8.6ms

**Throughput**
When we want to optimize for throughput, we ideally want to be compute bound, meaning we want to come close to utilizing all the TPU MXU capacity. Typically, that means increasing the batch size to be as large as possible so we are doing as much work as possible.

Repeating earlier sections, let's determine when a TPUv5e matmul becomes compute bound. This happens when time spent doing math (FLOPs), exceeds the time moving data from HBM into MXU.

Typically, we denote a matmul in this context as bf16[B, D] x bf[D, F]. Hence we get

T_math = 2BDF / C
T_comms = 2BD + 2DF + 2BF / (W_hbm)

Compute bound iff: T_math > T_comms <-> 2BDF/C > (2BD + 2DF + 2BF) / W_hbm <->
2BDF/(2BD + 2DF + 2BF) > C/W_hbm <-> {assuming B << D < F} B > C/W_hbm = 243

In BF16 the batch size needs to be larger than 243 to be compute bound. If our weights are int8 but FLOPs in bf16 then we will communicate half the bytes which means the necessary batch size is halved to 120. If we also perform our FLOPs in int8 these two factors cancel and we are back to 243. If our FLOPs precision is p_flops and our weight precision is p_w we can generalize the formula to be 

B > p_w/p_flops * C/W_hbm

The case of int8 weights and bf16 FLOPs is fairly common, since quantizing parameters losslessly is often easier than doing low precision arithmetic.

**Question:** What is the smallest TPU v5e topology we could serve LLaMA 3-70B on using bfloat16, int8, and int4 (both KVs and parameters) with 8k context? 
  
|dtype|param size|KV size / token (bytes)|min TPU v5es|actual min slice|remaining HBM for KV caches|num KV caches @ 8k|
|---|---|---|---|---|---|---|
|bf16|140GB|324kB|8.75|4x4 = 16 chips|116|43|
|int8|70GB|162kB|4.38|4x2 = 8 chips|68|52|
|int4|45GB|81kB|2.81|2x2 = 4 chips|19|67|
So in theory we can fit LLama 70B on just 4 chips, but with only 67 KV caches. Note that this is our batch size! That means we will have very poor utilization. Ideally we want to use a larger topology to push our batch size up to 240.

**Question:** Assume we use the largest batch size that fits on these topologies, what latency we could expect for each generate step?

Again we return to the latency equation from earlier 

$$\begin{align} \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\text{MLP (can be compute-bound)}} \end{align}$$
Where the MLP is compute bound if the token batch size is greater than B_crit. 

At BF16, B_crit is 240, and we are only memory bound. Which means the latency is

step latency = (Batch_size * KV cache + Parameter size) / Memory bandwidth = 19.2ms

We can alternatively realize that what we're doing here is just taking the total bytes that fit into TPU v5e HBM and moving those into MXU. That takes 

step latency = 16GB / 8.2e11 = 19ms

	Takeaway: we can always lower bound decode lateny by asking how long it takes to load all the model parameters from HBM into MXU. When KV caches are small, you can think about each layer as just loading the weights chunk-by-chunk and then discarding them. Unless we're using large batch sizes or lots of inter-device comms, this is often a reasonable bound. When our batch size is bigger, we need to model the KV cache as well, since that dominates the parameters. 

Likewise, in the FLOPs bound regime (e.g training or big-batch inference) the lower bound is determined by our FLOPs: 

Total FLOPs  / (N * C) = 2 * param count * B / (N * C)

where N is the number of accelerators sharded over, C the per accelerator flop. This is a lower bound, assuming no communication.

**Question:** For each of these, what throughput per chip does this give us (in terms of queries / chip)? _You can assume our median decode length is 512 tokens._

At BF16, our per step latency is 19ms. That means our throughput is given by

B / (per step latency * median steps * N) = 43/(0.019 * 512 * N) = 4.42/N. Plugging in N we get 

| dtype    | QPS / chip |
| -------- | ---------- |
| bfloat16 | 0.27       |
| int8     | 0.66       |
| int4     | 1.72       |
Doubling our topology would mean we can increase our batch size. With a 4x8 slice we now have 372GB over for KV caches, which means we can fit a batch size of 137. This means we get a throughput of 14/N. Giving

| dtype    | QPS / chip |
| -------- | ---------- |
| bfloat16 | 0.43       |
| int8     | 0.87       |
| int4     | 1.75       |
**Question:** Now let’s dig into the question of sharding. Let’s say we wanted to serve in bfloat16 on a TPU v5e 4x8. What sharding would we use for our model on a TPU v5e 4x8 during generation? Can we avoid being communication bound?

Right, so I think the first thing we want to look at is what happens if we apply TP. For a TPU v5e at bf16, TP is ici bound when Y > n_axis * F/2200 (the notes use 2550 which is for TPU v5p). If we shard across 2 axis, with our model config this is Y > 26. That means we can not TP shard across our entire slice without getting ICI bound. This means we can TP across a 4x4 slice but not a 4x8. And even this is generally optimistic since we rarely perfectly overlap communication. **Takeaway: we cannot actually serve on a 4x8 with pure model parallelism**. The best we can do is 4x2 or maybe a 4x4. 

## All about GPUs

**Question 1 [CUDA cores]:** How many fp32 CUDA cores (ALUs) does an H100 have? B200? How does this compare to the number of independent ALUs in a TPU v5p?

A H100 SM has 4 subpartitions, called SM subpartitions. Each subpartion contains a SIMD/SIMT vector architecture called a Warp Scheduler whose lanes (ALUs) calls CUDA cores. The CUDA Cores perform vector arithmetic, similarly to the TPU VPU, each ALU can generally do 1 arithmetic op each cycle, e.g a fp32 add. Each subpartition contains 32 fp32 cores (and a smaller number of int32 and fp64 cores). That means each SM has 128 fp32 CUDA cores. The H100 has 132 SMs, that means it has **16896 fp32 CUDA cores**. The B200 has 148 SMs / chip, **totalling 19536 CUDA cores**. The TPU v5p has 2 TensorCores, each wth a VPU with (8,128) lanes and 4 independent ALUs per lane so 2 * 4 * 8 * 128 = 8192 ALUs.

**Question 2 [Vector FLOPs calculation]**: A single H100 has 132 SMs and runs at a clock speed of 1.59GHz (up to 1.98GHz boost). Assume it can do one vector op per cycle per ALU. How many vector fp32 FLOPs can be done per second? With boost? How does this compare to matmul FLOPs?

Like we establishes, a H100 has 16896 fp32 CUDA cores (ALUs). It does one vector op per ALU per cycle. That means it does 16896 * 1.59e9 = 2.68e13 FLOPs per second. 3.34e13 in boost. In comparison, we can do 990 TFLOPs of matmuls in bf16. That is about 30x more FLOPs/s.

**Question 3 [GPU matmul intensity]:** What is the peak fp16 matmul intensity on an H100? A B200? What about fp8? _By intensity we mean the ratio of matmul FLOPs/s to memory bandwidth._

The intensity of a h100 is 1979e12/13.35e12 = 590. That is with sparsity. If we assume no sparsity it is 295. For b200 the fp16 is 2250e12 / 8e12 = 281. This means, in similar fashion to the TPU we need around 280 batch size to be compute bound in a matmul. For both the h100 and the b200 we have exactly 2x FLOPs in int8 compare to fp16. That means our peak intensity doubles to 590 and 562.

**Question 4 [Matmul runtime]:** Using the answer to Question 3, how long would you expect an `fp16[64, 4096] * fp16[4096, 8192]` matmul to take on a single B200? How about `fp16[512, 4096] * fp16[4096, 8192]`?

We expect `fp16[64, 4096] * fp16[4096, 8192]` to be comms bound. We can double check this to be exact.

T_comms = (2*4096*8192 + 2*64*4096 + 2*64*8192) / 8e12 = 8.58e-6
T_math = (2 * 64 * 4096 * 8192) / 2250e12 = 1.90e-6

We see that comms take longer than math, hence we are comms bound. Increasing the batch size to 512, we are now FLOPs bound, and the time is 15us. In both these cases we are calculating the theoretical LOWER BOUND. In reality we can expect to get a fraction of the maximum FLOPs and BW meaning the time is slower than calculated.

**Question 5 [L1 cache capacity]:** What is the total L1/SMEM capacity for an H100? What about register memory? How does this compare to TPU VMEM capacity?

The L1/SMEM is a small, very fast memory located inside each SM. This memory is the on-chip cache. It is programmer controlled. This is used for storing activations and inputs to the TensorCore matmuls. This is 256kb.

Each SM subpartition has its own register file containing 16384, 32-bit words, totalling 256 kib per SM. Basically, we have as much register memory as we have SMEM. In total this is 33MB of each per card. This is about half of modern TPU's VMEM.

**Question 6 [Calculating B200 clock frequency]:** NVIDIA reports [here](https://resources.nvidia.com/en-us-blackwell-architecture) that a B200 can perform 80TFLOPs/s of vector fp32 compute. Given that each CUDA core can perform 2 FLOPs/cycle in a FMA (fused multiply add) op, estimate the peak clock cycle.

B200 has 19536 CUDA cores, each performs 2 flops / cycle. We are performing 90TFLOPs per second, that means our cycle is 2.3GHz.

**Question 7 [Estimating H100 add runtime]:** Using the figures above, calculate how long it ought to take to add two `fp32[N]` vectors together on a single H100. Calculate both Tmath and Tcomms​. What is the arithmetic intensity of this operation? If you can get access, try running this operation in PyTorch or JAX as well for `N = 1024` and `N=1024 * 1024 * 1024`. How does this compare?

Adding two vectors fp32[N] means we have to load the two vectors into L1 cache, perform the operation in our cuda cores and then move the resulting fp32[N] vector back into DRAM.

T_comms = 12N / 3.35e12
T_math = N / 33.5e12

The arithmetic ratio of this operation is total FLOPs / total bytes = N/12N = 1/12 which is abysmal.

The peak hardware intensity is 10, meaning we are going to be horribly comms bound. That means the time of this operation is

T = max(T_comms, T_math) = T_comms = 12N / 3.35e12 = N/2.8e11. 

At N=1024 we expect

### Networking

differs a lot on GPUs and TPUs. As we've looked at thoroughly, TPUs are connected in 2D or 3D tori, where each TPU is only connected to its neighbors. That means sending data between two TPUs must pass through every intervening TPU and forces us to use only uniform communciation oatterns over the mesh. While this is inconvenient in some matters, it means the number links per TPUs is constant, and we can scale TPU "pods" to arbitrary sizes without loss of bandwidth. That's why we've been seeing examples with TPU pods up to 8192, and still using the same communication formulas. 

GPUs are different. They are connected in a tree-like hierarchy, where GPUs in a "node" are connected in an All-to-All fashion, within one hop of eachother, using high bandwidth interconnects called NVLinks. These nodes have typically consisted of 8 GPUs, but now exist up to 72 with the release of the GB200. A node historically referred to the NVLink domain and the number of GPUs connected to a single host, but since the GB200 they mean different things because the GB200 has a 72 NVLink domain but still only 8 GPUs per host. Nodes are connected into larger units (called SUs or Scalable Units) with a lower bandwdith InfiniBand or ethernet network. These in turn can be connected into arbitrarily alarge units with higher level switches. 

**At the node level**
NVLink connects GPUs within a node in an all to all fashion. These are upgraded together with the GPU generations (Ampere, Hopper, Blackwell). For the Hopper generation, NVLink bandwidth is 25 GB per link totalling 450 GB/s of full duplex bandwidth. This increases to 900 GB/s for blackwell. Comparing this to TPUs, the v5p and v6e have 180GB/s and the v5e has 90 GB/s. So, within each node GPUs are faster.

**Question 1 [Total bandwidth for H100 node]:** How much total bandwidth do we have per node in an 8xH100 node with 4 switches? _Hint:_ consider both the NVLink and NVSwitch bandwidth.

 Each NVSwitch has up to 64 ports, meaning it can handle up to 64 * 4 * 25e9 = 6.4TB/s. The DGX H100 configuration however only uses 18 Links per GPU, totalling 3.6TB/s.

**Question 2 [Bisection bandwidth]**: Bisection bandwidth is defined as the smallest bandwidth available between any even partition of a network. In other words, if split a network into two equal halves, how much bandwidth crosses between the two halves? Can you calculate the bisection bandwidth of an 8x H100 node? _Hint:_ bisection bandwidth typically includes flow in both directions.

Any even partition will have 4 GPUs in one half. Each of these halves can communicate 1800GB/s to the network, if we account for both sides communicating that makes 3.6TB/s of bisection bandwidth

**Question 3 [AllGather cost]**: Given an array of B bytes, how long would a (throughput-bound) AllGather take on an 8xH100 node? Do the math for bf16[D_x, F] where `D=4096`, `F=65,536`.

Okay, let's look at what each GPU is communicating in bytes. Each GPU holds a 2DF/X sized array which takes 

T_comms = 2DF/(X * W_uni) 

time to move. Because each GPU has the given unidirectional bandwidth (450GB/s) which we established earlier. Note that X * W_uni is the bisection bandwidth we calculated earlier. We need to perform (X-1) of these transfers.

T_comms = (X-1) * 2DF / W = 1.04ms

### Beyond the node level

Although the theoretical limit of GPU-to-GPU communication within a node is 450GB/s, if we actually look at empirical results we are unfortunately a bit away from this. In the image below Bus BW is the measured BW of our buses, which at at best closes in on 370GB/s. 

 ![[Screenshot 2025-08-25 at 19.31.47.png]]

In practice, this becomes a problem. Let's imagine we are performing an AllReduce over a reasonably sized array such as LLaMA-3 70B's MLP of shape bf16[8192,28672] only achieves around 150GB/s compared to the peak 450GB/s, this is 33% of the peak. Comparing this to TPUs, which are able to achieve their claimed BW at lower array sizes. 

![[Screenshot 2025-08-25 at 19.38.57.png]]

**In network reductions**
Since the Hopper generation, NVIDIA switches can now perform reduction operations **themselves**, and then multiplex or "MultiCast" the result to multiple target GPUs. This is called SHARP

A standard AllReduce (AllGather + ReduceScatter) cost w/o SHARP:

T_comms = 2 * B * (N-1) / (N * W_uni)

2x the cost of a normal AllGather. But, with SHARP. The data is sent to a switch, which performs the reduction and returns the result, halving the cost to

T_comms = B / W_uni

the same cost as a AllGather. In theory, this close to halves the cost of an AllReduce since it means each GPU can send its data to a top-level switch which itself performs the reduction and broadcasts the result to each GPU without having to egress each GPU twice. In practice however, SHARP provides about a 30% increase in BW, compared to the predicted 75%. 

![[Screenshot 2025-08-26 at 13.47.37.png]]

	Takeaway: in theory, NVIDIA SHARP should reduce the cost of an AllReduce on B bytes from about 2B/W to B/W. However in practice we only see roughly 30% improvement. Since pure AllReduces are fairly rare in LLMs, this is not especially useful.

#### Cross-node collectives
Collectives beyond the node level are a bit more subtle. Doing a reduction over a tree, we can think of as reducing from the bottom up, first within node, then leaf, then spine, using the normal algorithm at each level. 

To a first approximation, because we have full bisection bandwidth, the cost of an AllGather or ReduceScatter is roughly the buffer size in bytes divided by the node egress bandwidth *regardless of any of the details of the tree reduction.* This means our calculation looks similar to that of TPUs

T_{AG or RS comms} = bytes/W_{node egress} = bytes / 400e9 

This is possible because we are able to do a ring reduction over every node in the cluster. Beacuse of the fat tree topology, we can always construct a ring with W_node egress between any two nodes, and do a normal reduction. Let's dig into a more precise derivation of this, in the context of a full topology. In a full topology, we can imagine a reduction as performing a ring reduction at every layer in the network, which we can mostly overlap, giving us:

T_{AG or RS comms} = bytes * max_depth_i(D_i - 1 / D_i * W_link_i) 

where D is the number of children at depth i in the topology and W_link_i is the BW of the link connecting each child to node i. 

Using this we can calculate the available AllGather/AllReduce BW as 

min_depth_i(D_i * W_link_i / (D-1)) for a given topology. 

Using the case from before we have 

Node: D_node is 8 since we have 8 gpus in a node with a w_link of 450GB/s. Thuse we have AG bandwidth of 512GB/s.
Leaf: D_leaf is 32 since we have 32 nodes in an SU, with W_link = 400GB/s. This gives 413GB/s of BW.
Spine: D_spine = 4 since we have 4 SUs with W_link_i = 12.8TB/S. Our BW is hence 17.1TB/s. 

Hence our overall AG or RS BW is min(512, 413, 17.1) = 413GB/s at the leaf level. So in practice T_{AG or RS comms} = B/413GB/s, i.e we have about 413GB/s of AllReduce bandwidth even at the highest level. 

**Other collectives**. AllReduces are still 2x the above cost unless SHARP is enabled. AllToAll do change quite a bit cross-node, since they aren't hierarchical in the way AllReduces are. Let's remind ourselves what AllToAll collectives are. In AllToAll, we want to send each GPU a slice of an array. Because we all AllToAll connectivity, we can just send the shard directly to each GPU. Within a node, for B bytes, each GPU has B/N bytes. We send a 1/N of the array to N-1 targets. Meaning we communicate B/N^2 bytes, N-1 times:

T_comms = B * (N-1) / N^2 W ≈ B/NW 

Within a single node, this is a 1/8th theoretical speedup. Okay, let's go back to the cross-node situation. Unfortunately, like we were saying, AllToAll collectives are not hierarchical in the sense that reductions can performed layerwise. Instead AllToAll requires we communicate from every GPU to every GPU. If we want to perform an N-way AllToAll that spans M = N/8 nodes, on B bytes, each gpu holds B/M bytes and wants to send a 1/M shard of this giving


T_alltoall = B * (M-1) / M^2 * W_node_egress ≈ B / MW_node_egress.

That means we go from 

B/8 * 400e9 

within a single node to

B/2 * 400e9 

when we have 2 nodes. A 4x degredation. Scaling beyond a single node for the AllToAll collective essentially means we've got 50GB/s of bandwidth instead of 400:

B/N * 50e9 {N > 8, multi node}

**Collectives when an array is sharded over a separate axis**. When we want to perform a collective on an array which has a sharded axis which we are not reducing over, for example

AllReduce_x(A[Iy, J]){Ux}

we know that for TPUs this collective is cheaper, the overall cost is reduced by a factor 1/Y compared to the unsharded version because we are sending 1/Y as much data per axis. On GPUs, the cost depends on

**Pop Quiz 3 [Sharding along 2 axes]:** Say we want to perform AllGatherX(bf16[DX,FY])AllGatherX​(bf16[DX​,FY​]) where YY is the inner axis over a single SU (256 chips). How long will this take as a function of DD, FF, and YY?

We can break this into two cases. Where Y <= 8 and when Y>8. 

Y <= 8

bytes * (32 - 1) / (32 * )

Y > 8

2DF 256 / 

**Question 1 [SU AllGather]:** Consider only a single SU with M nodes and N GPUs per node. Precisely how many bytes are ingressed and egressed by the node level switch during an AllGather? What about the top-level switch?

Assuming an AllGather_x(A[Dx, F])

where the total bytes B = DF. 

Here we are assuming that the array is sharded over all GPUs, X=M * N. Meaning that each gpu has a B/X shard that they send to every other GPU: B/X * (X-1). 


#### Cross-node collectives  
  
As repetition, the cost to AllGather or Reduce scatter at the intra-node level of NVIDIA GPUS is given by the following. At the intra node level we have N gpus, B bytes. Each device wants to communicate B/N bytes. Due to the node setup we have direct connectivity between ALL devices in the node. That means each device wants to egress B/N bytes, to N-1 GPUs, and it can do that at the available GPU agress bandwidth. That means the cost of each hop is $T_{hop} = B/(N * W_{\text{gpu egress}})$  so the overall cost is  
$$  
T_{\text{intra-node AllGather or ReduceScatter}} = \frac{B * (N-1)}{(N * W_{\text{gpu egress}})} \approx \frac{B}{W_\text{gpu egress}}  
$$  
  
which you will note is the same as for TPUs. Similarly, the cost for an AllReduce is the combination of RS+AG, at twice the cost  
  
$$  
T_{\text{AllReduce}}  \approx \frac{2*B}{W_\text{gpu egress}}  
$$  
  
**AllGather and Reduce Scatter**  
  
Now, on-to **cross-node collectives**. When doing a reduction over a tree you can think of reducing bottom up, first within the node, then at the leaf level and then at the spine level. This has the nice effect that for an AllReduce, we communicate less data overall because we will reduce at the node level and we only have to egress $B$ bytes up to the leaf instead of $B*N$. Because we have full bisection bandwidth (the smallest bandwidth between any even partition of the network is equal to our full bandwidth) the cost of an AllGather or ReduceScatter is roughly the buffer size in bytes divided by the node egress bandwidth:  
  
$$  
T_{\text{cross-node AllGather or ReduceScatter}} \approx \frac{bytes}{W_\text{node egress}} = \frac{bytes}{400e9}  
$$  
You can imagine this as performing a ring reduction over every node in the cluster. Now you may be wondering, do we not have to perform the intranode reduction first, before we can do the cross-node reduction? Like often is the case, these two collectives are overlapped, and the intra node reduction will (almost) never be the bottleneck so we don't need to calculate it. But, the general cost is:  
  
$$  
T_{\text{total}} = \text{max}(T_\text{comms at node}, T_\text{comms in scale-out network}) = \text{max}[\frac{B}{W_\text{gpu egress}},\frac{B}{W_\text{node egress}}]  
$$  
  
**Precise calculation**  
  
Let's be even more precise in this calculation. As we've established, we're effectively doing a ring reduction at each layer in the tree (network) which we can mostly overlap. That means, the cost, is whichever reduction takes the longest. A general way to write this is   
  
$$  
T_{\text{AG or RS}} = B * \text{max}_\text{depth i}[\frac{D_i - 1}{D * W_\text{egress i}}]  
$$  
  
where $D_i$ is the degree at depth $i$, that is the number of children at depth $i$. To determine which level of the tree determines our time / BW, we just have to solve the max() part of the formula.  
  
Node: There are 8 GPUs with egress BW of 450GB/s, this will take 7 / (8 * 450e9) = 0.0019us  
Leaf: There are 32 nodes in an SU with egress BW of 400GB/s. This gives 31/(32 * 400e9) = 0.002us  
Spine: There are 4 SUs in total with egress BW of 12.8TB/s. This gives 4/(3 * 12.8e12) = 0.05ps  
  
As we can see, the bottleneck is at the leaf level.  
  
---  
  
**Other collectives**  
  
AllReduces are still 2x the above cost unless SHARP is enabled.   
  
AllToAlls change a bit in the cross-node because they are not hierarchical in the way AllReduces are. If we want to send data from every GPU to every other GPU we can't take advantage of the full bisection BW at the node level. That means if we have an N-way AllToAll that spans M = N/8 nodes, each node holds B/M bytes, it keeps 1/M and sends the rest to the other nodes (M-1). Giving  
  
$$  
T_{\text{cross-node AllToAll}} = \frac{\frac{B}{M} * (M - 1)}{W_\text{node egress}}  = \frac{B * (M - 1)}{M^2 * W_\text{node egress}} \approx \frac{B}{M * W_\text{node egress}}  
$$  
  
That means, when moving from a single node to two nodes, our AllToAll collectives go from $B / (8 * 450e9)$ to $B/(2 * 400e9)$.  A general formulation of this is:  
  
$$  
T_{\text{AllToAll}} =  
\begin{cases}  
\displaystyle \frac{B}{N * W_{\text{gpu egress}}}, & N \leq 8, \\[1.2em]  
\displaystyle \frac{B}{W_{\text{node egress}} \cdot \tfrac{N}{8}}, & N > 8.  
\end{cases}  
$$  
  
which for our full fat tree is  
  
$$  
T_{\text{AllToAll}} =  
\begin{cases}  
\displaystyle \frac{B}{N * 450e9}, & N \leq 8, \\[1.2em]  
\displaystyle \frac{B}{N * 50e9}, & N > 8.  
\end{cases}  
$$  
  
    Takeaway: beyond the node level, the cost of an AllGather or ReduceScatter on B bytes is roughly B/W_node egress, which is B/400e9 on a H100 DGX SuperPod.  
  
**Reductions when array is sharded over a separate axis**  
  
In TPU-land, performing reductions such as  
  
$$  
\text{AllReduce}_X (A[I_Y,J](U_X))  
$$  
  
where we reduce over an array that has a dimension sharded over a separate axis **reduced the cost by a factor 1/Y**.  This makes sense because we are moving 1/Y less data in each hop. Unfortunately, in GPU-land, this is not as straight forward. On GPUs, the cost depends on which axis is the "inner" one (intra-node vs inter-node) and whether each shard spans more than a single node. Going back to the general formulation  
  
$$  
T_{\text{total}} = \text{max}(T_\text{comms at node}, T_\text{comms in scale-out network})  
$$  
First, look at the intra node setting
$$  
T_{\text{intra-node}} = \frac{B}{W_{\text{gpu egress}}} * \frac{1}{\min(Y, D)}  
$$  
Then the scale out  
  
$$  
T_{\text{scale-out network}} = \frac{B}{W_{\text{node egress}}} * \frac{D}{\max(Y, D)} 
$$  
Case 1: $Y < D$

$$  
T_{\text{node}} = \frac{B}{W_{\text{gpu egress}}} * \frac{1}{Y}  
$$
$$  
T_{\text{scale-out network}} = \frac{B}{W_{\text{node egress}}}
$$
$$  
T_{\text{scale-out network}} > T_{\text{node}} \leftrightarrow \frac{B}{W_{\text{node egress}}} > \frac{B}{W_{\text{gpu egress}}} * \frac{1}{Y} \leftrightarrow Y > \frac{W_{\text{node egress}}}{W_{\text{gpu egress}}} = 0.88
$$
which is always true meaning that the scale out time dominates when our shards spans less than a node. 

Case 2: $Y > D$

$$  
T_{\text{node}} = \frac{B}{W_{\text{gpu egress}}} * \frac{1}{D}  
$$
$$  
T_{\text{scale-out network}} = \frac{B}{W_{\text{node egress}}} * \frac{D}{Y}
$$
$$  
T_{\text{scale-out network}} > T_{\text{node}} \leftrightarrow  \frac{B}{W_{\text{node egress}}} * \frac{D}{Y} > \frac{B}{W_{\text{gpu egress}} * D} \leftrightarrow Y < \frac{W_{\text{gpu egress}}}{W_{\text{node egress}}} * D^2 = 72
$$
which means that the scale out term dominates as long as we shard across less than 72 GPUs. Up to that point, our time will decrease proportional to 1/Y, meaning we ideally want to shard across 72 GPUs. After that, the intra node term takes over, which does not depend on Y, and further Y sharding is not beneficial.

**Quiz 4: Collectives**  
  
**Question 1 [SU AllGather]:** Consider only a single SU with M nodes and N GPUs per node. Precisely how many bytes are ingressed and egressed by the node level switch during an AllGather? What about the top-level switch?  
  
Let's work through the components of the reduction.   
  
Each GPU holds B/NM bytes of data. Within each node, each GPU sends B/NM to the switch, for a total ingress of BN/NM = B/M bytes ingressed.  
  
The switch egresses B/M bytes to the spine switch.  
  
We ingress B(M-1)/M bytes from the spine switch.  
  
We egress B - B/MN bytes N times for a total of BN - B/M