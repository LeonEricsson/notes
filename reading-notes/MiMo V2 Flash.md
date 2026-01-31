### arch
The model follows a fairly standard MoE architecture, with hybrid attention, interleaving Sliding Window Attention and GQA. There are 5 SWA blocks, followed by a single global attention block, this structure is repeated 8 times. RMS normalization is used throughout, in pre-norm style. The MoE layers have 256 expert with 8 activated per token, no shared experts, an activation ratio of 3.1%. 

![[Screenshot 2025-12-22 at 22.38.09.png|500]]

mimo uses the same learnable attention sink bias as gpt-oss, noting that it enhances performance of hybrid SWA models dramatically, matching global attention baselines. An ablation of attention variants on a 32B model shows that across general benchmarks (MMLU, GSM8K, MATH), a Hybrid SWA with a window of 128 with attention sink outperforms GA. The same can be said when looking at long context benchmarks such as NoLiMa, RULER-32K, and at more compelx reaosning benchmarks like AIME24/25, LiveCodeBench. A window of 128 performs better than W = 512. 

### pretrain
Mimo V2 Flash is pretrained on 27T tokens. The data processing pipeline has a deliberate shift toward data exhibiting long-range dependencies. Pre-training is split into three sequential stages:

1. *Stage 1 (Pre-training, 0 - 22T)* the model is trained on diverse, high quality general-purpose corpus using a context of 32K tokens to establish strong foundational language capabilities.
2. *Stage 2 (Mid-training, 22-26T)* the data mixture upsamples code-centric data and incorporates 5% synthetic reasoning data to enhance logical reasoning and program synthesis abilities.
3. *Stage 3 (Ctx extension, 26-27T*) Following the stage 2 data distribution, the models context window is extended to 256k tokens and data is usampled with long-range dependencies. 

The model is trained with AdamW (b1=0.9, b2=0.95) and a weight decay of 0.1 Stage 1 pretraining is conducted at 32k context. At stage 3 this extends to 262k.

### posttrain
post training is conducted in three stages, starting with supervised fine-tuning used to transform the base model inte a helpful assistant capable of following instructions and responding effectively across diverse tasks. The SFT dataset spans millions of sampels across general conversation, reasoning, coding and agent tasks. The samples collected are from both thinking and non-thinking modes. 

Following SFT, is RL.

#### RL
RL is separated into two distinct categories, non-agentic and agentic RL, each with its own strategy. Non-agentic RL is single turn tasks that don't require interactive feedback or multi-step execution, in these domains the primary task is to enhance the model's reasoning accuracy in verifiable domains while remaining helpful and safe. For these tasks the team uses programmatic reward verifiers paired with LLM judge. For subjective qualities they use rubric based frameworks.

in agentic training, the RL model operates in interactive, multi-turn environments that requires planning, action execution and adaption based on feedback. agentic RL training is scaled across two key dimensions: environment diversity and compute. Code agent, terminal agent, web dev agent and general agent are trained separately. For each domain they develop robust large scale environments. Training the coding agent across 120K environments improves eprformance on SWE