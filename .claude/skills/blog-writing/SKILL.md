---
name: blog-writing
description: Transform blog drafts (unstructured research notes) into polished, publication-ready technical blog posts. Use when given a draft file to convert into a finished post.
---

# Blog Draft to Post Transformation

Transform raw research notes and draft compilations into polished technical blog posts. Drafts contain gathered information—notes from reading papers, code snippets, math formulations—that need to be synthesized into a cohesive narrative with consistent style and tone.

## When to Use This Skill

Use this skill when:
- Converting a draft file in `blog/drafts/` into a finished post
- The user provides research notes to turn into a blog post
- Refining an existing blog post draft

**Key principle**: Drafts are raw material, not blueprints. The draft contains rough notes, information gathering that may be beyond the scope of the downstream blog-your job is to curate, synthesize, and craft a narrative, not transcribe.

---

## Core Philosophy

### For Researchers, By a Researcher

Blog posts target readers with **high technical competence**. Assume familiarity with:
- Standard ML/DL concepts (transformers, attention, gradient descent, reinforcement learning)
- Mathematical notation and derivations
- Code literacy (Python, PyTorch)

**If a prerequisite isn't explained in the draft, don't explain it in the post.** The draft author knows what their audience knows. Trust that judgment.

### Voice and Tone

The writing voice is **direct, professional, casual, and humble**—someone with genuine love for this work, without being overbearing.

**Characteristics:**
- First person ("I posit", "I've been following", "Let's see how")
- Direct statements, not hedged claims
- Enthusiasm that's understated, not performative
- Technical precision without pedantry
- Conversational flow without sacrificing rigor

**What to avoid:**
- Academic stuffiness ("In this work, we present...")
- Excessive hedging ("It may perhaps be the case that...")
- Hype language ("groundbreaking", "revolutionary")
- Over-explanation of basics
- Sycophantic openings ("Large language models have achieved remarkable success...")

### Style Reference Examples

Study these passages for tone calibration:

> "To explain batch invariance, let’s simplify the system and look solely at matmuls. You can assume that all matmul implementations are “run-to-run deterministic."This is not totally true, but most common matmul implementations do have this property. However, they are not “batch-invariant.” In other words, when the batch size changes, each element in the batch can get different results"

> "- This is a fairly unusual property from a mathematical perspective. Matrix multiplication should be “independent” along every element in the batch — neither the other elements in the batch nor how large the batch is should affect the computation results of a specific element in the batch"

> "I've been a big fan of the RL wave that has drenched the LLM space since the release of DeepSeek-R1. Seeing the level of personal experimentation that this has enabled has been fascinating to follow..."

> "Having a mental model of what a transformer is and how it operates helps you quickly absorb new findings, providing an anchor to latch onto and integrate new learnings. A mental model isn't static; while some parts will be held with greater conviction—especially when solidified by repeated findings—the overall model must continuously shift and adapt."

> "To understand this intuitively, imagine teaching a child to aim at a target. If they shoot too far to the left, you'd tell them to adjust right; too far right, adjust left. The size of the adjustment depends on how far they missed—a concept directly reflected in the Delta Rule."

> "This linear, additive structure of the residual stream has some really cool and important consequences."

**Pattern**: Technical depth + accessible analogies + genuine curiosity + direct voice.

---

## Common AI-Generated Writing Failures

When transforming drafts, actively avoid these patterns that make posts read like documentation rather than authored understanding:

### Encyclopedia Voice

**Problem**: Opening with definitions or declaring what a technique "is."

**Bad**: "Geometric Sequence Masking is a technique for importance sampling correction..."

**Good**: "Distribution shift between training and inference frameworks was a recurring theme in late 2025. For whatever reason, I've found myself drawn to this problem..."

The reader should feel they're learning alongside someone who has worked through the material, not receiving a Wikipedia entry.

### Missing Personal Stakes

**Problem**: No sense of *why this author cares* or *why now*.

Establish:
- What prompted the investigation (a paper, a bug, a conversation)
- Personal engagement with the topic ("I scan every new paper for...")
- Temporal context when relevant ("over the later half of 2025...")

### Declarative Claims vs. Offered Perspectives

**Problem**: Titles and framing that declare facts rather than offer viewpoints.

**Bad title**: "X is Y" (declarative claim)
**Good title**: "a lens for X" or "understanding X through Y" (offered perspective)

Humility about your framing invites engagement; declarations shut down dialogue.

### Options Without Reasoning

**Problem**: Listing alternatives without explaining when/why to choose each.

**Bad**: "Two approaches exist: TIS clips ratios; MIS discards them."

**Good**: "As usual with RL, the choice *depends*. Clipping treats every sample as fundamentally usable and interprets extreme ratios as a variance problem. But extreme ratios typically arise when... Masking takes the opposite stance..."

Show the *reasoning* about choices, not just the options.

### Math Without Meaning

**Problem**: Showing equations without interpreting what they mean.

After every significant equation, add an interpretation paragraph. Ask: "What is this equation really asking?" and answer in plain English.

**Bad**: [equation] followed immediately by code or next section

**Good**: [equation] → "This decomposition clarifies what OPSM is really asking: 'is this negative sample still representative of what the current policy might produce?'"

### Missing Practical Implications

**Problem**: Presenting techniques as abstract facts without connecting to real use cases.

Add speculation and connection to trends:

"My guess is that hard trust region enforcement becomes especially important as trajectories grow longer with agentic work and complex reasoning problems..."

### Disconnected Sections

**Problem**: Sections that read like independent encyclopedia entries.

Use explicit causal transitions:

- "So far we've covered X... But this only addresses one aspect."
- "This leads naturally to the question of..."
- "With this foundation, we can now understand why..."

### No Uncertainty Markers

**Problem**: Presenting one framing as definitive truth.

Include:
- "My guess is..." for speculation
- A caveat section acknowledging alternative approaches
- "Still too early to tell..." when appropriate

Intellectual honesty about limitations builds trust.

### Reference-Style Scaffolding

**Problem**: Numbered sections, tables of contents, formal "Summary" sections—signals documentation.

Use simple lowercase headers. Let prose do the summarizing. End with insight, not enumeration.

### References as Footnotes

**Problem**: Parenthetical citations disconnected from narrative.

**Bad**: "AReaL (https://...) proposed this"

**Good**: "I first encountered this in AReaL... but it really came into focus after Fengyao's blog post"

References become part of the discovery story—they're witnesses to how understanding evolved.

### Unplanned Notation

**Problem**: Using generic notation that becomes confusing when similar concepts appear later.

Plan notation for the *entire* post before writing. If you'll have two geometric means, distinguish them from the start ($\rho_{\text{geo}}^{\text{TI}}$ vs $\rho_{\text{geo}}^{\text{OPSM}}$).

---

## Transformation Workflow

### Step 1: Understand the Draft

Before writing anything:
1. Read the entire draft to understand the full scope
2. Identify the **core insight** or **central claim**
3. Note which sections are background vs. novel contribution
4. Look for explicit notes from the author (e.g., "[needs more]", "FOR BLOG:", "TODO")
5. Identify what can be cut—drafts often contain tangential material

**Ask**: What's the one thing a reader should take away?

### Step 2: Design the Narrative Arc

A blog post is a **story**, not a dump of information. The reader should feel they're learning alongside someone who has worked through the material—not receiving a reference document.

**Discovery narrative structure:**

1. **Personal hook**: Why does *this author* care? What prompted the investigation? ("For whatever reason, I've found myself drawn to this problem...")
2. **Problem setup**: What's broken or missing? Frame it as a real challenge.
3. **Existing approaches**: What tools already exist? Why are they insufficient?
4. **Core insight**: The aha moment. What resolves the tension?
5. **Deeper understanding**: Connect insights, show implications, add speculation.
6. **Honest limitations**: What's still uncertain? What are alternative framings?

**Key principle**: Structure reveals how understanding was *built*, not just what was learned. The progression should feel causal: "This led me to try X, but X has problem Y, which motivates Z."

### Step 3: Blend Modalities

Posts should, when applicable, **fluently maneuver** between:

1. **Prose explanations**: Build intuition, tell the story
2. **LaTeX formulations**: Mathematical precision, formal definitions
3. **Code examples**: Concrete implementations, executable understanding

This blend conveys competence and serves different reader preferences. Some readers skim to the math; others want code they can run; others want the prose narrative.

**Guidelines:**
- Introduce concepts in prose *before* showing the math
- After a complex equation, provide intuition or a code implementation
- Code should be minimal and focused—not full implementations
- Use `$$...$$` blocks for display math

**Example flow:**
```
[Prose: explain the intuition behind importance sampling]
[Math: show the formal definition]
[Prose: interpret what the ratio means]
[Code: simple implementation showing the calculation]
[Prose: discuss implications]
```

### Step 4: Write with Consistent Style

**Sentence-level guidelines:**
- Keep subject and verb close
- Put emphasis at sentence ends (stress position)
- One paragraph = one point
- Use active verbs, not nominalizations ("We analyzed" not "We performed an analysis")

**Terminology:**
- Pick consistent terms and stick with them
- Define notation once and use it throughout
- Don't switch between equivalent terms (pick "importance weight" or "likelihood ratio", not both)

**Formatting:**
- Use headers to create clear sections
- Use blockquotes for direct quotes from papers
- Tables for comparisons

**Bold and italics**: Use deliberately, not scattered throughout. Bold is appropriate for key term definitions on first use or critical warnings. Italics for emphasis on a specific word or for technical terms. If every paragraph has bold text, none of it stands out.

**Lists and bullets**: Overuse creates choppy, disjointed reading that obscures rather than clarifies. Lists are effective for genuine enumerations (steps, options, items) but often indicate missing logical flow and structure. When you reach for a bullet list, ask: would prose with proper transitions serve better? Usually yes. A paragraph that connects ideas with "First... This leads to... Therefore..." reads better than the same content chopped into bullets.

### Step 5: Self-Contained Completeness

The post must **stand alone**. A reader shouldn't need to reference external material to understand the core argument (though they may want to for deeper dives).

- Define all notation used
- Provide enough context for the central claim
- Link to references for background, but don't require them

---

## Quality Standards

### What Makes a Great Post

1. **Clear central thesis**: One sentence should capture the core insight
2. **Logical flow**: Each section follows naturally from the previous
3. **Appropriate depth**: Deep enough to be useful, not so deep it loses focus
4. **Mathematical precision**: Equations are correct, notation is consistent
5. **Working code**: Any code shown should be correct and runnable
6. **Honest limitations**: Acknowledge what you don't know or didn't cover

### Red Flags to Avoid

- Wall-of-text paragraphs without breaks
- Code blocks without context
- Math without intuition
- Sections that don't connect to the central thesis
- Explanations of obvious prerequisites
- Hedging on every claim
- Hype or overselling
- Bold/italics scattered throughout every paragraph
- Excessive bullet lists where prose would flow better

---

## Frontmatter Format

Blog posts use this YAML frontmatter:

```yaml
---
layout: post
title: "Post Title"
categories: [RL, LLM, etc]
year: 2025
type: blog
---
```

---

## Draft Conventions

Drafts may contain special markers:

| Marker | Meaning |
|--------|---------|
| `[needs more]` | Section is incomplete, may need research |
| `[FOR BLOG:]` or `[NOTE:]` | Explicit instruction for the final post |
| `[TODO]` | Author reminder, may need attention |
| `> quote` | Direct quote from a paper—cite appropriately |
| `---` | Section break in notes |

When you encounter these:
- `[needs more]`: Decide if the section is essential. If yes, flag it; if no, cut it.
- `[FOR BLOG:]`: Follow the instruction directly
- `[TODO]`: Evaluate if it's blocking or can be resolved

---

## Transformation Checklist

Before delivering a finished post:

```
Narrative:
- [ ] Central thesis is clear and stated early
- [ ] Each section serves the narrative
- [ ] Logical flow from setup to conclusion
- [ ] Appropriate scope (cut tangents)
- [ ] Explicit transitions between sections (causal, not just sequential)

Technical:
- [ ] All math is correct
- [ ] Notation is consistent throughout AND planned for full post
- [ ] Code examples are correct and minimal
- [ ] References cited appropriately

Style:
- [ ] First-person voice, direct and humble
- [ ] No hedging or hype
- [ ] Prose/math/code blend is smooth
- [ ] Consistent terminology

Authorial Presence (anti-AI-slop):
- [ ] Opens with personal context, not encyclopedia definition
- [ ] Establishes why this author cares / why now
- [ ] Every significant equation followed by interpretation
- [ ] Alternatives presented with reasoning, not just listed
- [ ] Includes uncertainty markers where appropriate ("My guess is...")
- [ ] References integrated into narrative, not just cited
- [ ] No formal summary section at end (let prose conclude)
- [ ] Title frames perspective, not declarative claim

Format:
- [ ] Frontmatter is complete
- [ ] Headers create clear structure (simple, not numbered)
- [ ] Images referenced correctly (`![](/assets/...)`)
- [ ] Internal links use wiki-link format if needed
```

---

## Example: Draft to Post

**Draft excerpt:**
```
### PPO
Proximal Policy Optimization (PPO) introduces a clipped surrogate objective:
$$J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip}...$$

Here, $\pi_\theta(a|s)$ is the current policy...

[needs more on why this matters for LLMs]
```

**Post version:**
```
The ratio $\frac{\pi_\theta}{\pi_{\text{old}}}$ is the heartbeat of PPO. It tells us:
how much more (or less) likely is this action under our current policy compared to
when we collected this data?

$$
J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A,
\text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)},
1-\varepsilon, 1+\varepsilon\right)A\right)
$$

This correction enables multiple gradient steps per batch—crucial for LLM training
where generating rollouts is expensive and we want to squeeze as much learning as
possible from each generation pass.
```

**What changed:**
- Added interpretive prose before the math
- Connected to the LLM context (addressing the [needs more] note)
- More conversational framing
- Same technical content, better narrative flow

---

## Example: Encyclopedia Voice → Discovery Narrative

**Encyclopedia voice (bad):**
```markdown
# Geometric Sequence Masking

Geometric Sequence Masking is a technique for importance sampling correction
in RL training that addresses a fundamental problem: when your inference
engine and training framework produce slightly different distributions,
standard sequence-level importance weights become length-biased.

## 1. Background: The PPO Ratio

PPO's clipped surrogate objective contains a ratio between policies...
```

**Discovery narrative (good):**
```markdown
# a lens for understanding X

Distribution shift between training and inference frameworks was a recurring
theme in late 2025. For whatever reason, I've found myself drawn to this
problem. Every time a new paper drops, I scan for any mention of
training-inference mismatch.

There are two paths forward. You can eliminate the discrepancy at the source...
Or you can accept that some mismatch is inevitable and correct for it
algorithmically. The latter has been far more common, and is the focus of
this post.

I'll walk through the standard tools for handling off-policy data, show why
naive approaches break down for long sequences, and arrive at geometric
sequence masking as a length-invariant alternative. This all leads somewhere
specific...

## background

PPO's clipped surrogate objective contains a ratio between policies...
```

**Key differences:**
- Opens with temporal/personal context, not definition
- Establishes author's relationship to the topic
- Previews the journey (problem → tools → failure → solution → insight)
- Uses lowercase headers, no numbering
- Title offers perspective ("a lens for") rather than declaring fact
