# syndiff

**syndiff** is a lightweight, standalone library that implements a structural diff algorithm inspired by **Difftastic**, designed to be embedded inside other tools and applications.

It compares code as *syntax*, not just text - producing diffs that align with how developers reason about changes.

### ✦ Key features

* **Structural diffs** - Operates on parsed syntax trees instead of lines.
* **Embeddable core** - Designed as a library, not a CLI application.
* **Language-agnostic** - Parsers live at the edges; the algorithm stays generic.
* **Deterministic output** - Predictable, explainable diffs suitable for tooling and UIs.
* **Minimal surface area** - Focused api intended for reuse and experimentation.

### ✦ Philosophy

Traditional diffs treat code as text. Syndiff treats code as having *shape*.

This library exists to make syntax-aware diffing easy to integrate into editors, code review tools, language servers, and research projects - without dragging along a full application.

It does not aim to replace Difftastic. It distills its core ideas into a reusable form.

### ✦ Inspiration

Inspired by **[Difftastic](https://github.com/Wilfred/difftastic)**, created by Wilfred Hughes. All credit for the original algorithmic ideas belongs there. Any bugs introduced here are new and entirely our own.
