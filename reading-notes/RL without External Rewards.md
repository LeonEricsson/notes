
RLVR while requiring little human supervision still requires gold annotations and domain-specific verifiers. In mathematics, this requires expert annotation of solutions; in code generation, it necessitates comprehensive test suites and execution environments.

To sidestep this issue and find a more general and scalable reward paradigm the authors ask

> *Can LLMs enhance their reasoning abilities by relying solely on intrinsic, self-generated signals, without recourse to external verifiers or domain-specific ground truth?*

There is a line of research establishing that LLMs exhibit lower confidence on difficult problems. Using self-certainty, the average KL divergence between the model output distribution and the uniform distribution, as a confidence measure, the authors propose optimizing directly for confidence, using the models confidence as an intrinsic reward signal. This completely eliminates the need for external supervision or handcrafted rewards. 

The authors apply this to Qwen 2.5 family through GRPO with self confidence acting as the reward signal. There has been doubts recently on Qwen's suitability for such tasks, apparently you can train Qwen models on broken rewards and it still improves. 