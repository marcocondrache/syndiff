# syndiff

**syndiff** is a lightweight, standalone library that implements a structural diff algorithm inspired by **Difftastic**, designed to be embedded inside other tools and applications.

It compares code as *syntax*, not just text - producing diffs that align with how developers reason about changes.

### ✦ Key features

* **Structural diffing** - compares syntax trees, not text lines
* **Embeddable by design** - a library core, no CLI or UI assumptions
* **Language-agnostic** - parsing is external; the algorithm is generic
* **Deterministic output** - stable, predictable diffs for tooling
* **Minimal API** - small surface area, easy to integrate and experiment

### ✦ Philosophy

Traditional diffs treat code as text. Syndiff treats code as having *shape*.

This library exists to make syntax-aware diffing easy to integrate into editors, code review tools, language servers, and research projects - without dragging along a full application.

It does not aim to replace Difftastic. It distills its core ideas into a reusable form.

### ✦ Inspiration

Inspired by **[Difftastic](https://github.com/Wilfred/difftastic)**, created by Wilfred Hughes. All credit for the original algorithmic ideas belongs there. Any bugs introduced here are new and entirely our own.
