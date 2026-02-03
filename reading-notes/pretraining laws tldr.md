

#### kaplan (chinchilla) scaling laws - dense transformers

The paper estimates total compute to be approximately three times the cost of a single forward pass: $C_\text{train} ≈ 3 * C_\text{fwd}$. For a dense model, a general rule of thumb is that the number of FLOPs in the forward pass is `2 * num_parameters` per token. This gives a simple rule $C_\text{train} ≈ 6 * N * D$, where N is the number of parameters, and D the number of training tokens. 

Using this approximation, the fitted scaling laws derived in the paper are:

$$
N_{\mathrm{opt}}(C) \approx 0.145 * C^{0.49}  
$$
$$ 
D_{\mathrm{opt}}(C) \approx 1.15 * C^{0.510}  
$$
Famous takeaway:
$$
D_{\text{opt}} ≈ 20N
$$

##### examples

**Train a 1.3B model according to chinchilla compute-optimal frontier**
 
Eliminate $C$ from the scaling laws to get $D$ as a function of $N$:  
$$  
D_{\text{opt}}(N) \approx 1.15\left(\frac{N}{0.145}\right)^{0.510/0.490}  
 = 
1.15\left(\frac{N}{0.145}\right)^{1.040816}  
$$  
Plugging $N = 1.3e9$:  
$$  
D_{\text{opt}} \approx 1.15\left(\frac{1.3e9}{0.145}\right)^{1.040816} = 2.63\times 10^{10}\ \text{tokens} \approx 26.3\ \text{B tokens}  
$$
Roughly the same as using the takeaway $D_{\text{opt}} ≈ 20N = 26 \text{B tokens}$.

**Academic pretraining 1.3B for 100B tokens**

Typical academic pretraining scale is $N=1.3$B  for $D=100$B tokens. How many more tokens than optimal is this?

$$  
D_{\text{opt}}(N=1.3e9) \approx 1.15\left(\frac{1.3e9}{0.145}\right)^{1.040816} = 26.3e9 
$$

$$
\implies \frac{D}{D_{\text{opt}}}=\frac{100}{26.3}\approx 3.80\times  
$$


So it’s about 3.8× the optimal tokens.

