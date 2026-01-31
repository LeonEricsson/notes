DeepSeek's own hosted API models and website have seen decreasing traffic since its release, despite the hype around the model. People are using DS models through other providers, why?

Comparing DeepSeek R1 $/Mtok through their own API vs other hosters we see that DeepSeek fails to outcompete in things like latency, and context window.

![](https://i0.wp.com/semianalysis.com/wp-content/uploads/2025/07/image-34.png?resize=1385%2C810&ssl=1)
*Source: [https://openrouter.ai/](https://openrouter.ai/) accessed in May 2025. Blended $/Mtok calcuated with 3:1 input:output ratio, bubble size represents context window size*

So there are a lot of better options out there if you want to use DS models. This is an active decision by DeepSeek: by batching more users simultaneously on a single GPU or cluster of GPUs, the model provider can INCREASE the total wait experienced by the end user with higher latency and slower interactivity to DECREASE the total cost per token. Higher batch sizes and slower interactivity will reduce the cost per token at the expense of a much worse user experience. DeepSeek are not interested in making money off users, they are minimizing the necessary compute to serve these models to users, preserving GPUs for research. 

Batching at extremely high rates allows them to use the minimal amount of compute possible for inference and external usage.

