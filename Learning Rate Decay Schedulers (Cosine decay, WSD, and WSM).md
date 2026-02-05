Typically, learning rate schedulers consisted of two phases: an initial warmup phase followed by a decay. During the warm-up, the LR increases linearly from a small value to a peak value, $lr_{peak}$, over $T_\text{warmup}$ steps.  This helps stabilize the optimization process in the early stages of training. Following warm-up, the LR gradually decreases according to a predefined function, such as cosine, linear or inverse square root decay. 
Typically, learning rate schedulers consisted of two phases: an initial warmup phase followed by a decay. During the warm-up, the LR increases linearly from a small value to a peak value, $lr_{peak}$, over $T_\text{warmup}$ steps.  This helps stabilize the optimization process in the early stages of training. Following warm-up, the LR gradually decreases according to a predefined function, such as cosine, linear or inverse square root decay. 

**Standard Cosine Decay**
The conventional Cosine schedule reduces the learning rate from the peak to a minimum value over a fixed duration. While effective, it rigidly requires the total training steps, $T_{max}$, to be defined a priori. The decay phase is formulated as:

![[Pasted image 20260204125130.png]]

This dependency on $T_{max}$ means that extending training requires a full restart to recalibrate the decay curve. 

**Warmup-Stable-Decay (WSD)**
To mitigate this rigidity, WSD inserts a stable phase with a constant learning rate between the warmup and decay phases, maintaining $lr_{peak}$, for a number of steps. This allows the decay to be initiated at an arbitrary step $T_{decay\_start}$.  The scheduler is formulated as:

![[Pasted image 20260204125143.png]]

Which means we no longer need to manual tune the LR during the stable phase, and we have multiple options for decay attempts from the endpoint. This also has the added benefit of not having to reset at the initial state, we can reset to  $T_{decay\_start}$ if we want to continue training . However, despite this added flexibility, we will need to predefine decay phase settings, such as when to start, the decay function, and how many total training steps to go for. 

**What is happening during the decay?**
One might wonder, why are we decaying in the first place, and what is it about the decay that forces us to reset to a pre-decay checkpoint if we want to continue pretraining? Well, the decay phase is a way to force models to settle into a sharp local minimum to maximize performance on the current data. As LR approaches zero, the model weights "harden" or "ossify". During high LR the model has increase plasticity, and explores the loss landscape freely. In [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective](https://arxiv.org/abs/2410.05192) the authors describe the global loss landscape as a river valley. During the stable phase the model is free to traverse the length of the river, but as the decay phase starts, the model is carefully moving down the valley walls, in a very local optimization landscape. The WSM authors propose a similar analogy: "*constant LR training resembles traversing a “canyon” with oscillating steps, while merging [or LR decay] resembles finding a “river” at the canyon floor that guides efficient converge*". Once the decay phase has ended, it becomes very difficult to safely move out of this local minimum to learn new things without damaging what it already knows. This is often referred to as a drop in plasticity. Taking the model from a decayed state, and you try to return to traversing the length of the river by cranking the learning rate back up the model experiences a massive chock which has been empirically observed to create loss spikes and catastrophic forgetting. In the MiniCPM paper they explicitly state that trainign from a decayed checkpoint results in poor performance. They recommend reverting to a checkpoint from the stable phase, arguing that the stable phase maintains the model in a "fluid" state suitable for continued learning, whereas the decay phase "freezes" the model into a specific data mixture.

### WSM
To minimize scheduling complexity, we ideally want to do away with the decay phase from LR schedules. Recent work has shown success in weight averaging (a.k.a model merging) techniques as a prominent direction, where empirical results demonstrate that simply maintaing a constant LR combined with standard weight average strategies can achieve performance competitive with WSD-based schedules.

Warmup-Stable and Merge (WSM) builds on this research direction by formalizing a connection between LR decay and checkpoint merging. Through this they demonstrate that WSM can be instantiate to emulate various decay strategies- including cosine decay, linear decay and inverse square root decay.  WSM simplifies the LR schedule to:

![[Pasted image 20260204150245.png]]

The central hypothesis of WSM is that the optimization benefits of LR decay can be decoupled from the live training process and instead be effectively achieved through the merging of model checkpoints. This allows you to train with a high learning rate (high plasticity) while simultaneously extracting a model that behaves as if it were trained with a decaying learning rate (high stability/convergence), simplifying both continued pretraining, and required LR schedule configuration.

### **The Mathematical Equivalence of Merging and Decay**

The core theoretical contribution of WSM is proving that **checkpoint merging** is mathematically equivalent to training with a **gradient decay schedule**. This allows you to train with a high learning rate (high plasticity) while simultaneously extracting a model that behaves as if it were trained with a decaying learning rate (high stability/convergence).

**Theoretical connection between LR Decay and Checkpoint Merging**
Let $\theta_{n}$ be the model parameters at step $n$. We have a sequence of $k+1$ checkpoints: $[\theta_n, \theta_{n+1}, ..., \theta_{n+k}]$. The merged model $\hat{\theta}_{n+k}$ is a weighted average of these checkpoints:

$$\hat{\theta}_{n+k} = \sum_{j=0}^k c_j \theta_{n+j}$$

where $c_j$ are non-negative weights summing to 1 ($\sum_{j=0}^k c_j = 1$). Assuming standard SGD-like updates where $\theta_{i+1} = \theta_i - g_i$ (where $g_i$ is the gradient update vector including the learning rate), any future checkpoint $\theta_{n+j}$ can be written as the initial checkpoint $\theta_n$ minus the sum of all gradient vectors:

$$\theta_{n+j} = \theta_n - \sum_{l=1}^j g_{n+l-1}$$
By plugging the gradient expansion into the merging formula, we get:

$$\hat{\theta}_{n+k} = \sum_{j=0}^k c_j \left( \theta_n - \sum_{l=1}^j g_{n+l-1} \right)$$

The key mathematical trick here is swapping the order of summation. Instead of summing over checkpoints $j$, we sum over the gradient steps $i$. A gradient $g_{n+i-1}$ affects every checkpoint $\theta_{n+j}$ where $j \ge i$. This yields:

$$\hat{\theta}_{n+k} = \theta_n - \sum_{i=1}^k \left( \sum_{j=i}^k c_j \right) g_{n+i-1}$$

The term inside the parenthesis acts as an effective weight for that specific gradient step. We define this effective weight as $w_i$:

$$w_i = \sum_{j=i}^k c_j$$

This gives us the final equivalent form:

$$\hat{\theta}_{n+k} = \theta_n - \sum_{i=1}^k w_i \cdot g_{n+i-1}$$

This equation reveals that merging checkpoints is equivalent to applying a **synthetic decay schedule** defined by $w_i$ to the gradients. Even though the live training used a constant learning rate, the merged model effectively "sees" the gradients scaled down by $w_i$. At this point, you have to approximate decay functions with $w_i$, and then calculate the resulting checkpoint weights. I don't fully understand how this works, the theorem provided in the paper:

![[Pasted image 20260205075003.png|700]]

But you can see some examples here of decay functions and their corresponding checkpoint merging weight distributions

![[Pasted image 20260205075128.png|700]]

