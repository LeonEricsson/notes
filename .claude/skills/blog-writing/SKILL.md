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

A blog post is a **story**, not a dump of information. Structure around:

1. **Hook**: Why should the reader care? What problem or curiosity drives this?
2. **Setup**: Minimal background needed to understand the contribution
3. **Core content**: The main insight, derivation, or technique
4. **Synthesis**: What does this mean? How does it connect?
5. **Takeaway**: Crisp summary of the key insight

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

Technical:
- [ ] All math is correct
- [ ] Notation is consistent throughout
- [ ] Code examples are correct and minimal
- [ ] References cited appropriately

Style:
- [ ] First-person voice, direct and humble
- [ ] No hedging or hype
- [ ] Prose/math/code blend is smooth
- [ ] Consistent terminology

Format:
- [ ] Frontmatter is complete
- [ ] Headers create clear structure
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
