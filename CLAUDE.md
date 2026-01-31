# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an Obsidian vault containing personal notes on ML/AI research, primarily focused on large language models, transformers, and reinforcement learning.

## Structure

- `reading-notes/` - Paper summaries with YAML frontmatter (layout, title, categories, year, type, author, exturl)
- `blog/` - Longer-form technical posts and synthesis pieces
- `general/` - Reference docs and evolving research notes
- `assets/` - Images referenced via `/assets/filename.png`

## Frontmatter Format

```yaml
---
layout: post
title: "Paper Title"
categories: [NLP, RL, etc]
year: 2024
type: paper  # or: blog
author: LastName
exturl: https://arxiv.org/...
---
```

## Content Conventions

- LaTeX math via `$$...$$` blocks
- Images: `![](/assets/image.png)`
- Internal links: `[[Note Title]]` (Obsidian wiki-links)
- External links: standard markdown `[text](url)`
