

### Olmo 3 Base
is a strong foundational model released at 7B and 32B. The models are dense models with a standard decoder-only Transformer architecture. With regards to the architecture there are very little interesting deviations from the norm, you've got QK-Norm, GQA and Sliding Window Attention on 3/4 layers. Training is performed at bf16 and achieves 43% MFU during training. 

#### Pretraining

![[Screenshot 2025-11-30 at 17.57.13.png]]

The main "case" for Olmo is their complete transparency in training data all the way to model weight release. The authors release multiple data mixes for pre-training, mid training, RL, and so on, including model checkpoints during training as well as a base model, an instruct model, and an RL-ed model. The pretraining datamix is called **DOLMA 3 MIX** and is made up of similar data sources to other pretraining recipes. The pipeline for data curation looks like

![[Screenshot 2025-11-30 at 18.32.08.png]]

#### Long-context extension
A typical part of "mid-training" is extending context. Across open-source recipes this is performed to drastically different extensions, some recipes perform long-context training for only a few billion tokens, with others training for as much as 1T long context tokens. There is also no concensus on when to apply long context extension training. Llama aplies it prior to mid training, Qwen afterwards and GLM 4.5 after SFT. Olmo 3 uses long documents from the olmoOCR Science PDF pool. 

### Olmo RL
The algorithimic details are what you'd expect if you've been following open source RL work this year. It's almost an exact replication of the work that has made its circles in twitter with very little deviations unlike a lot of previous RL work. I suspect this is because Olmo as opposed to a lot of other RL papers, actually started their RL runs later in the year after the whole GRPO popcycle had finished. The objective is essentially a combination of DAPO and Dr GRPO combined with truncated importance sampling. 

**Verifiers**. To perform RLVR they use different verification systems for each domain. For math they use a rule-based verifier that performs basic normalization and compares with reference anser using SymPy. For code they use a test-case based verifier that runs a set of test cases over the response. For chat they use a LM judge paired with rubric grading. 

#### Infra
Substantial work was put into improving the RL infrastructure to handle longer sequences and faster overall throughput. Inference dominated the costs during RL training, using 8 H100 nodes for training and 20 noes for inference for the 32B olmoRL reasoner model. They employ a off-policy asynchronos RL setup with a centralized learner distributed across notes and a large pool of independent actors running vLLM. The learner produces prompts that are queued and sipatched to the actors, which execute the prompts, interact with the environment and return results through a results queue that the learner uses to update the model parameters. 

