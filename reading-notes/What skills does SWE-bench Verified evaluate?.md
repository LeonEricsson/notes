A post from EpochAI. EpochAI keeps putting out really solid stuff, their benchmarks are great, their analysis are great. Strongly support their work.


SWE-Bench Verified is one of, if not the most prominent agentic coding benchmarks. SWE-Bench is a collection of real-world software issues sourced from GitHub. The benchmark involves giving agents a code repository and issue description, and challenging them to generate a patch that resolves the problem described by the issue. The -Verified subset is a human-validated subset of SWE-Bench released by OpenAI in August 2024.

SWE-Bench Verified is 500 Python-only coding problems, evaluated using unit tests. The problems are relatively simple, solvable within 1 hour for a human engineer. Saturation on this benchmark would be strong evidence that models are capable of autonomously fixing small to medium issues in actual real-world Python repositories. This isn't just implementing a single function, like LiveCodeBench or Aider Polygot, this is agentic, end-to-end solutions of problems.

The model is given:
- The repository state before the PR was merged
- The issue description.

You then evaluate the model by running any tests, including additional tests that were introduced in the ground truth PR.

This selection process means that the type of problems represented in SWE-Bench are somewhat biased compared to real-world coding. In SWE-Bench, the problem is well-contained, and the solution is unambiguously defined. This is not representative of the messy reality many engineers face.

#### The error rate in SWE-Bench Verified is relatively low
EpochAI analyse the hardest problems in the benchmark, and manually handpick a selection of these to try and determine which problems contain actual systematic errors, which make them unsolvable. One of the most important aspects of benchmarks is the error rate, i.e., the percentage of tasks that are either incorrect, ambiguous, or impossible to solve. Epoch determine this rate to be between 5% and 10%.

#### Most tasks are simple bug fixes
While the error rate is low, it does not measure models capability to solve GitHub issues in general: around 90% of the tasks in the benchmark are fixes that experienced engineers could complete in under 1 hour. Therefore, the benchmark really tests **whether AI can make simple codebase edits.**  Most (87%) of the issues are classified as bug fixes, with only 9% being feature requests. 

The dominance of small issues can be explained by the curation process of the benchmark. Filtering for merged PRs that also change the test suite heavily favors small, isolated bugs.

#### The low diversity of codebases limits external validity
The benchmark is sourced from only **12 repositories**, and even within these the distribution is very skewed. 

![[Screenshot 2025-07-03 at 20.04.27.png]]

#### The samples are old
The benchmark was created in 2023, with over half of the samples being from before 2020. Therefore, it is safe to assume that all current and future models have likely been trained on the relevant codebases, their respective changes over time and even the issues themselves, as these public repositories are an important source of training data.