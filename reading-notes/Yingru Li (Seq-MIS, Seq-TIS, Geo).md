
## Trust Regions
### Soft vs. Hard Trust Regions

The TRPO framework suggests two approaches to enforce trust regions:

| Type                 | Mechanism                              | Implementation                      |
| -------------------- | -------------------------------------- | ----------------------------------- |
| **Soft (Clipping)**  | Down-weight samples outside the region | sample included with bounded weight |
| **Hard (Rejection)** | Exclude samples outside the region     | sample excluded entirely            |

**Soft trust regions** use clipped importance sampling. This is computationally efficient but retains potentially problematic samples.
**This part develops hard trust region methods**—Seq-MIS and Geo-Mask—that completely exclude samples outside the trusted region. We show when and why hard rejection outperforms soft clipping.

#### Problem: When rejection outperforms clipping
The implicit assumption when performing clipping on sequence-level importance ratios is that all samples provide a valid learning signals - samples with high weights simply require variance control via clipping. However, in practice this assumption fails and there are indeed extreme ratios that arise when our sampling distribution is near the numerical precision floor (u(y) ≈ 10^-9). These are OOD samples which may occur due to numerical precision artifacts, distribution shifts etc. The problem with clipping here is that the samples are still in the gradient update which may introduce sytematic errors into every gradient step. 

#### Solution: Hard Trust Region via Rejection
Instead of soft clipping we enforce a **Hard Trust Region** where samples outside the region are rejected entirely. Only samples within the region contribute to the gradient, samples with p(y) > C are treated as unreliable and excluded. 

**Seq-TIS**: Should be used under moderade mismatch; maximize sample efficiency
**Seq-MIS**: Should be used under large mismatches; when OOD samples are likely and you want to prioritize robustness.

The choice depends on the reliability of high-weight samples. When the behavior policy  is well-calibrated and mismatch is controlled, Seq-TIS extracts more information. When OOD samples are prevalent, Seq-MIS provides a Hard Trust Region that prevents gradient corruption. In practice, monitoring the rejection rate provides insight into the policy-behaviour divergence and can guide the choice between estimators. 

## Length-Dependent Rejection Bias
For autoregressive generation, the sequence-level importance ratio is a product of per-token ratios

![[Screenshot 2026-01-06 at 12.29.02.png]]

Even when the per-token mistmatch is small, this product grows exponentially with sequence length T **almost everywhere** along the trajectory. If the mean log ratio p != 1, for example say that the ratio is slightly above 1 at 1.001 (0.1% per-token drift) we end up with a length dependent acceptance probability:

![[Screenshot 2026-01-06 at 12.31.08.png]]

which is a systematic length bias. For any fixed threshold C there exists a critical length T* beyond which almost all samples are rejected. Reasoning models often generate sequences containing thousands of tokens, which means that with just slight per-token drift we reject almost all long reasoning chains, meaning the model receives systematic biased feedback favoring short outputs. There is no single fix for this, instead you are faced with a trade-off where a large threshold C >> 1 allows you to accept long sequences but introduces high variance and allows OOD sampels while a small threshold controls variance and OOD samples but systematically biases against long sequences. There is no value C that both: accepts high-quality long reasoning chains (requires C large for when T is large), rejects low quality or OOD samples (requires C small enough to filter outliers) , provides a length invariant acceptance criterion (requires C to adapt to length T).

This motivates the need for a length-invariant trust region mechanism - which is exactly what Geometric Sequences Masking provides.

## Geometric Sequence Masking
The fundamental problem of a sequence-level importance ratio is that $p(y) = \sum_t p_t$ is an **extensive quality** - it scales with sequence length. We can not control such a quality with a fixed region. We need an **intensive quality** that measures the average per-token divergence, independent of length. 

![[Screenshot 2026-01-06 at 12.41.37.png]]

This is the geometric mean of the per-token ratios. It is length-invariant: if every $p_t = r$, then $p_{geo} = r$, regardless of T.

Within the geometric ratio, we can define a **Per-Token Trust Region** that is independent of sequence length: 

![[Screenshot 2026-01-06 at 12.44.38.png]]

## Summary: Hierarchy of Estimators and Selection Guidelines

![[Screenshot 2026-01-06 at 12.51.04.png]]

For long-horizon reasoning tasks, Geometric Sequence Masking (Geo-Mask) provides a principled, length-invariant Hard Trust region that prevents the systematic length bias inherent in standard importance sampling estimators. 