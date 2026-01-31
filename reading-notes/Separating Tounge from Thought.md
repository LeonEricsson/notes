This paper builds on the findings by Wendler et al in *Do Llamas Work in English? On the Latent Language of Multilingual Transformers* by analyzing whether models use language-agnostic concept space when processing multilingual text, primarily on the task of translation. This analysis is performed through a method known as activation patching. Where one systematically replaces activations of one forward pass with activations from another to identify how and where the model thinks.  
  
#### activation patching  
  
How does the LLM "know" that the Eiffel Tower is in Paris? What parts of the network are crucial for retrieving and using this factual information? To start we need 2 prompts.  
  
1. **Clean Prompt:** A prompt where the landmark and its correct city are present or implied.  
    - "The Eiffel Tower is in the city of"  
2. **Corrupted Prompt:** A minimally different prompt where the landmark is the same, but we expect a different (incorrect) city  
    - "The Colosseum is in the city of" (We expect "Rome").  
  
**The Activation Patching Experiment:**  
  
1. **Run the Clean Prompt:** Run "The Eiffel Tower is in the city of" and record the activations at various layers and token positions, especially around the token "Tower" and when the model predicts "Paris".  
      
2. **Run the Corrupted Prompt:** Run "The Colosseum is in the city of" and observe the model predicting "Rome".  
      
3. **Patch Activations:** Take the activations from a specific layer and token position (e.g., the activation after processing "Tower") in the **clean run** and insert (patch) them into the corresponding layer and token position during the **corrupted run** (after processing "Colosseum").  
      
4. **Observe the Output:** After patching, run the corrupted prompt _with the patched activation_ and observe the model's next token prediction.  
      
**aha!**  
  
If patching the activations from the "Eiffel Tower" run into the "Colosseum" run causes the model to now predict "Paris" (or significantly increase the probability of "Paris" compared to "Rome"), it suggests that the patched activation at that specific layer and token position carries information crucial for associating "Eiffel Tower" with "Paris".  

  
#### task prompt design  
  
For a given concept C, input language ℓ(in), and output language ℓ(out), we construct a few-shot translation prompt TP(ℓ(in), ℓ(out), C). This prompt contains examples of single-word translations from ℓ(in) to ℓ(out), concluding with the model being tasked to translate C from ℓ(in) to ℓ(out). For example, TP(EN, FR, CLOUD) could be:   
  
```  
English: “lake” - Français: “lac"   
...   
English: “cloud" - Français: “  
```  
  
Here the task is to translate w(CLOUD^EN ) = {“cloud"} into w(CLOUD^FR ) = {“nuage"}.  
  
Importantly, whether the model correctly answers the prompt is determined by its next token prediction.  
  
This lets us track the probability of a concept occuring in a certain language $l$, by simply summing up the probabilities of all tokens in $w(C^l)$ in the next-token distribution. Intuitively this allows us to track how *well* a language dependent concept is encoded at a certain stage of the model. Compared to Wendler et al. who only considered a single possible translation, this method considers all possible expressions of $C$ in $l$.  
  
### initial explanatory analysis  

#### anthropic recent work  
  
Anthropic's recent interpretability using sparse autoencoders on very similar translation tasks show that models during translation are driven by very similar *circuits*, with shared multilingual components and an analogous language specifc component. Visualized below   
  
![[Pasted image 20250528152902.png]]  
  
Anthropics result came after this paper, in March of 2025.   
  
---  
  
The aim of this paper is very similar, but approaches the problem through a completely different approach, that of activation patching, attempting to find similar results to Anthropic. So the results from this paper are very interesting.  
  
**The aim is to understand whether language and concept information can vary independently during Llama-2's forward pass when processing a multilingual prompt.** Specifically, imagine a that the model holds a representation of $C^l$ as $z_{C^l} = z_c + z_l$ where $z_c \in U$ and $z_l \in U^\perp$ and $U + U^\perp = R^d$  is a decomposition of $R^d$ in a subspace $U$ and its orthogonal complement $U^\perp$. Such a representation would allow for language and concept information to vary independently: language can be varied by changing $z_l$ and concept by changing $z_c$. Conversely, if language and concept information are not decomposably according to this, then varying the aforementioned representations would result in a "failure", meaning the output is not what is expected, varying the concept would vary the language and vice versa. Hopefully at this point, it is becoming clear how activation patching may come into play here.

The initial experiments are designed to infer at which layers in the model the output language and concept enter the residual stream of the next token position $h_{n_T}^j(T)$ and whether they can vary independently of the task. 

Two datasets are formed, the source $S$ and target $T$, both containing a translation task with a concept, input language and output language. A sample looks as previously described above in Example A. For each pair of source and target two parallel forward passes are performed, and for each transformer block the residual stream of the last token of the source prompt is patched into the corresponding layer (block) of the target prompt forward pass by explicitly setting $h_{n_T}^j(T) = h_{n_S}^j(S)$ and then allowing the rest of the forward pass to continue unaltered. Then, one also records the resulting next token distribution of the source language, source concept, target language, target concept $P(C_S^{l_s})$, $P(C_S^{l_T})$, $P(C_T^{l_s})$, $P(C_T^{l_T})$. 

At some point during the forward pass, the residual stream of the last token will have to encode information of the output language and concept. This analysis builds on the assumption that observing resulting next token distributions is a measure of at what point this move happens. Now it could indeed be the fact that language and concept are transfered in stages, e.g over multiple layers, and in that case we would not be able to see a distinct change in the downstream next token distribution. So let's look at what the results are:

![[Screenshot 2025-05-29 at 11.33.53.png]]
*Our first patching experiment with a DE to IT source prompt and a FR to ZH target prompt with different concepts. The x-axis shows at which layer the patching was performed and the y-axis shows the probability of predicting the correct concept in language ℓ (see legend). In the legend, the prefix “src" stands for source and “tgt" for target concept. The orange dashed line and blue dash-dotted line correspond to the mean accuracy on source and target prompt. We report means and 95% Gaussian confidence intervals computed over 200 source, target prompt pairs featuring 41 source concepts and 38 target concepts.*

So, let me try to break down what's happening here. We have a source concept, and a source input + output language, and the same for target. Remember we are patching into the target prompt forward pass. When patching in during the first 12 layers, the resulting distribution is basically unchanged, it still "correctly" predicts the target concept in the target language chinese. Then, between layers 12-16 things start to get interesting, we see that the probability for the target concept remains high but now the language is no longer Chinese but rather Italian, which is the output language of the source sample. Moving forward, starting at layer 17 we note that the model predicts the source concept in the source language. Wow! It seems as if the output language is the first thing that gets moved to the residual stream, around layer 14 then independent of the language we see that the concept is moved over to the residual stream at layer 17.  If these two things were entangled, we shouldn't see the probability of the source language together with the target prompt rise while the source concept probability still being very low. This seems to be indicate that these things are somehow encoded in separate orthogonal spaces of the residual stream. Another thing I find interesting is thinking about what happens at layer 13, where the model is just as likely to predict Italian as it is Chinese, this seems to indicate that the model moves / transforms the language subspace of the residual stream over multiple layers, becoming slightly more confident throughout the layers of the forward pass 12 - 14. 

From these observations we form the following hypthesis:

**H1**. Concepts and language are represented independently. When doing translation, the model first computes $l^{out}$ from context, then identifies $C$. In the last layers, it then maps $C$ to the first token of $w(C^{l^{out}})$.

**H2**. The representation of a concept is always entangled with its language. When doing the translation, the model first computes $l^{out}$ , then computes $l^{in}$ and $C^{l^{in}}$ from its context and solves the language-pair-specific translation task of mapping $C^{l^{in}}$ to $C^{l^{out}}$ .

--- 

So, the next step becomes trying to rule out one of the hypothesis. In the above experiments we never observed the source concept in the target language. However both H1 and H2 would allow for that to happen via patching in the right way. 

