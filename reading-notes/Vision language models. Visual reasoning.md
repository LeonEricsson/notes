
#### QvQ

Follow up model to QwQ (which is a text-based reasoning model) with inference time scaling similar to that of o1. Unfortunately there are no details on the training of QvQ. You send the model a picture and a prompt and it reasons about the problem, you can not send a follow up prompt.

#### SmolVLM

This is a smol release from Huggingface, attempting to scale down the size of VLMs while retaining as much performance as possible. Released in April 2025 we're still seeing the same overarching approach: patchify -> vision encoder -> linear layer / cross modal adapter -> llm.
The llm in use, SmolLM2, has a limited context window of only 2k which is a problem when combining it with a vision encoder because they the 'vision tokens' use a lot of tokens. Hence, the LLMs are fine-tuned on longer context data increasing their capacity to 16k (and 8k for the smoller variants). To further reduce the 'vision tokens' the authors perform *pixel shuffling* which rearranges spatial features into additional channels, reducing spatial resolution but increasing resolution density. Pixel shuffle is applied after the vision encoder and before the cross modal adapter.

![[Screenshot 2025-04-11 at 09.31.47.png]]

#### InternVL3

The latest release in the InternVL family and it's an exciting one! Combining separately trained vision encoders and pretrained LLMs has been the defacto standard approach in vision language models for the past year, year and a half. Funny that just a few weeks ago we saw the "Scaling Laws for Native Multimodal Models" paper arguing that natively multimodal architectures provide better performance tradeoffs compared to these standard "late-fusion" approaches. Now, InternVL3 drops and its just that, a natively trained VLM. 

InternVL3 uses a *native multimodal pre-training* approach that consolidates language pre-training and multi-modal pretraining into a single pre-training stage. This differs from previous InternVL versions that tried to bridge the gap between the two modalities by intricate parameter freezing or multi-stage fine-tuning schedules to ensure core linguistic capacities remain uncompromised.  

>Unlike conventional paradigms—where a language-only large model is first trained (typically with language pre-training followed by language post-training) and subsequently adapted to accommodate additional modalities—our method performs integrated optimization by interleaving multimodal data (e.g., image–text, video–text, or interleaved image–text sequences) with large-scale textual corpora during the pre-training process

Now, while they do perform native multimodal pretraining, the actual components are still **initialized** from existing vision encoders and language models. The architecture remains largely the same so what's really changed here is the data, the type of data being trained on and in what order.

Overall the architecture of InternVL3 follows the same framework as its predecessors with "ViT-MLP-LLM". The vision encoder comes in two configurations: InternViT-300M for the smaller family members, and InternViT-6B for the larger. As for the language model they choose models from both the Qwen2.5 series and InternLM3 8B. The MLP is two layers, randomly initialized.

**Variable Visual Position Encoding** is a new position encoding scheme which increments the position index for visual tokens less than for text tokens. Normally, the position index of are incremented by 1 in line with the tokens, so token $x_i$ has position index $i$. But V2PE increments visual tokens with a number $\delta < 1$. 

**The multimodal autogressive formulation.** 

### Visual Reasoning

#### Vision-R1

First attempted to directly apply R1-Zero style training to a VLM by collecting a dataset of 10K math problems. However the training failed and the authors deemed this straight forward approach to not be feasible under the constraints of data quality and quantity.

Instead, the authors perform a cold-start using a multimodal CoT dataset. The dataset is generated automatically, without human annotation, by first prompting a VLM to describe the image in text, performing what the authors call a **modality bridging**: 

> *Given a image, a question:{question} and a thinking process:{thinking process}, provide a detailed description containing all the necessary details of the image to answer the question correctly...*

Then, after cleverly bridging image information to textual information, the problem is fed into DeepSeek-R1 to solve. After filtering out R1s failure attempts you end up with the high-quality multimodal CoT dataset called Vision-R1-cold which is used to cold start initialize Vision-R1. After cold-start Vision-R1 is trained with GRPO in R1-Zero fashion. 

**Benchmarks.** Vision-R1 is trained on visual math problems from datasets such as WeMath, MathVision, Polymath, ... . This is somewhat interesting but it is a quite narrow domain. The benchmarks used are MathVista, MathVerse and MM-Math.

#### R1-Zero Training for VSI

This paper takes on the task of improving visual spatial reasoning of models through a R1-Zero style training setup. Visual spatial reasoning is the act of perceiving, remembering and reasoning about spatial information obtained through visuals. VSI-Bench, released on the tail end of 2024, is a benchmark designed to evaluate the visual-spatial intelligence of Multimodal LLMs. VSI-Bench comprises over 5,000 question-answer pairs derived from 288 egocentric videos sourced from the validation sets of public indoor 3D scene reconstruction datasets.

![[Pasted image 20250411111408.png]]

**VSI-Bench** includes eight tasks categorized into three types: configurational, measurement estimation, and spatiotemporal. Performance on this benchmark is far below that of human performance, as can be seen by their own evaluation leaderboard.

https://github.com/vision-x-nyu/thinking-in-space?tab=readme-ov-file

![[Pasted image 20250411111512.png]]

To improve visual-spatial reasoning, the authors construct a video-based question answering dataset VSI-100k and then train Qwen 2 VL using GRPO. They use format and accuracy rewards, in a typical fashion.  LoRA training, 14 rollouts per question, KL beta 1e-4. 

The results are decent, with Qwen 2B achieving an average accuracy of 35.4 and Qwen 7B achieving 40.7.  Notably, the authors find that a vanilla system prompt 

> *Please provide the short answer with one or two words or a number.*

works significantly better than something like a *think* system prompt. They also find that the KL penalty is necessary to avoid training instability.  Finally they compare performance to DPO and SFT, finding that GRPO outperforms both. Notably, this paper reports **no issues in training with pure RL from a VLM base model** as opposed to the Vision-R1 paper. 


### Benchmarks

#### MMMU 
*A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert
AGI* according to the paper. This benchmark is the equivalent of MMLU but for multimodal models. It covers a wide variety of subjects, and aims to evaluate how well models not only perceive and understand information but also apply reasoning with subject specific knowledge. It is important to understand that MMMU **requires** college-level knowledge in combination with strong reasoning.  

![[Screenshot 2025-04-11 at 11.42.51.png]]

which may or may not be what we want.

#### MEGA-Bench
MEGA-Bench contains 505 multimodal tasks with diverse data sources, input/output formats, and skill requirements. This benchmarks requires a diverse and wide set of reasoning abilities but similar to MMMU it also requires a certain degree of world knowledge. 

#### MathVista, MathVerse
Math based visual benchmarks that require strong reasoning capabilities but purely within math. This places less emphasis on the actual visual understanding / perception and more so on the ability to convert the problem from a visual one to a textual one such that it can be solved formally. 

#### Visual Spatial Reasoning
The Visual Spatial Reasoning corpus is a collection of caption-image pairs with true/false labels. Each caption describes the spatial relation of two individual objects in the image, and a vision-language model needs to judge whether the caption is correctly describing the image or not. It is a very old benchmark and is most likely close to being saturated. I have only seen this referenced in PaliGemma paper.

### VLM Mech Interp

#### Probing
Used to identify if features are encoded in the models intermediate representation. Simple linear probes are preferred. A high accuracy linear probe suggests the property is well-encoded in the models internal representation. However it only shows correlation, not causation. Just because the property is encoded doesn't mean it is used to produce the final output. 

Used to show modality prioritization, VLMs prio language over vision.

#### Activation Patching
Selectively modify internal activations, while keeping other constant, and analyzing downstream performance to identify which components contribute to certain model behaviour. 
How-to:

1. **Save Activations:** Record the internal activations of a model when processing clean and corrupted inputs.
2. **Select Target Activations:** Identify the specific activations to modify.
3. **Patch Activations:** Replace activations from one input (e.g., corrupted) with those from another (e.g., clean).
4. **Rerun the Model:** Run the model with patched activations and observe behavioral changes.
5. **Analyze Results:** Infer the role of specific components based on how the output changes.

Used to show 

- that representations at visual token positions gradually align with interpretable textual concepts as one descends the network layers. 
- that LLaVA primarily processes broader contextual information in the early layers and specific object details in the later. 

#### Logit Lens
Apply the unembedding layer to intermediate layers, to project the activation into vocabulary space. Allows one to analyze intermediate activations in terms of their output representation. Sort of shows internal thinking. However the logit lens may be misleading as it is a projection on the vocabulary space, and may in fact contain mostly information that is orthogonal to vocabulary space. 

Used to show
- Activations in the late layers at each visual token position correspond to token embeddings that describe the patch object.

#### SAE
Sparse Autoencoders act similarly to linear probes, but are designed to disentangle complex internal representations that occur inline with the superposition hypothesis. The superposition hypothesis poses that neurons in a underdimensioned network are "forced" to represent multiple overlapping features, making direct interpretation very difficult. SAEs tackle this by taking internal activations and mapping them to a higher dimensional sparsely activated space, enabling the extraction of distinct interpretable features.

Have yet to be applied to VLMs.

****