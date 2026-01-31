
Everyone knows that long context benchmarks have become more of a PR stunt than actual performance metrics at this point. Models are being sold as having a certain context lengths, with lengths up to 1M being advertised, but with effective (read usable) ctx lengths being significantly shorter.

Evaluating a models context boils down to evaluating a distinct two stage process:

1. **Recall / Retrieval.** Find the relevant segment(s) in the text.
2. **Reasoning.** Use the recalled text to perform the task at hand. 

The difficult of **both** of these stages is crucial to developing a proper long context benchmark. The history of long context benchmarking has primarily focused on making stage 1 of this process difficult, starting with the now classic Needle-in-a-Haystack benchmark. From NIAH, benchmarks evolved to introducing multiple needles, distractor context, and better masking the needled in the context, but ultimately, this still focuses only on the retrieval part of long context understanding. Benchmarks like RULER, a slightly more difficult NIAH variant, which are still being used for major model releases have saturated, because at this point models are good enough **to access their entire context window**. Having a look a sample from RULER we see how "easy" the benchmark is compared to real world long context tasks 

	...Which means that what matters is who you are, not when you do it. If you're the right sort of person, you'll win even in a bad economy. And if you're not, a good economy won't save you. The special magic number for XXX is 12345. Someone who thinks "I better not start a startup now, because the economy is so bad" is making the same mistake as the people who thought during the Bubble "all I have to do is start a startup, and I'll be rich."...

	Question: What is the special magic number for XXX?

The needle is completely OOD from the rest of the context, so this task is purely about finding (retrieving) the correct part of the context, we're just testing if the model can access its own context. [NoLiMa](https://arxiv.org/abs/2502.05167) is the first variant of NIAH that introduces som level of reasoning; the needle is a piece of information that comes from the same distribution as the haystack but with minimal literal overlap with the query, forcing at least one step of reasoning to draw association between the two. 

OpenAI's MRCR is another example of a retrieval heavy long context benchmark. 
#### designing a long context benchmark

long context capabilities become increasingly important as models improve task [time-horizon](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) exponentially. With the length of tasks AI can solve doubling every 7 months, they need to be able to properly recall **and** reason over their entire context window. Increasing the difficulty of both of these stages, make for a good long context benchmark.  

A good example of such a benchmark is LoCoDiff which provides models with git diffs and asks them to output the final state of the file. In this case, there is no relevancy check as every diff is relevant. However, the task difficulty does scale with the sequence length as the model has to handle more state changes. Another good example is Fiction.live which is based on a set of stories: each sample is a question about one of the stories. Answering the questions requires having a theory of mind for the characters, an understanding of the chronology of events and an ability to make inferences based on implicitly stated information. This also requires reasoning over large parts of the context implicitly to derive a final state of the characters in the story. 

nrehiew's LongCodeEdit is a recent example of a good long context benchmark. The benchmark uses samples from BigCodeBench which provides python functions with complete docstrings. Said functions are combined to fill the context window and then a single function is manually corrupted to ensure they fail the provided test case. The corruptions are minor, and subtle, requiring the model to search through the entire context window while it reasons about each function independently in terms of its logic w.r.t its given docstring.
#### concluding thoughts

this isn't to say that benchmarks evaluating either stage 1 or stage 2 are bad, understanding what aspects of long context understanding models fail in is very important, but if we want to emulate real world long context tasks, we need to make both aspects difficult. 