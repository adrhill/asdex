# Documentation Guide for AI Agents

## Diataxis Framework

The docs follow the [Diataxis](https://diataxis.fr/) framework.
Each page belongs to one of four categories:

- **Tutorials** (`tutorials/`): Learning-oriented.
  Guide a beginner through a complete experience to build skills.
  Show the goal upfront, deliver visible results early and often.
  Minimize explanation — let the reader learn by doing.
  Don't offer choices or alternatives; keep the path narrow and reliable.
- **How-To Guides** (`how-to/`): Task-oriented.
  Give directions to solve a specific real-world problem.
  Assume the reader already has basic knowledge and knows what they want to achieve.
  Don't teach or explain — just show the steps.
  Write from the user's perspective, not the tool's.
- **Explanation** (`explanation/`): Understanding-oriented.
  Explain concepts, design decisions, and theory.
  No step-by-step instructions.
- **Reference** (`reference/`): Information-oriented.
  Auto-generated from docstrings via mkdocstrings.
  Keep docstrings accurate and complete.

## Semantic Line Breaks

All markdown content uses **semantic line breaks**:
one sentence or clause per line.
This makes diffs cleaner and is a firm requirement.

Bad, avoid:
```markdown
Graph coloring assigns colors to vertices such that adjacent
vertices get different colors. This allows computing multiple
rows in a single AD pass.
```

Good:
```markdown
Graph coloring assigns colors to vertices such that adjacent vertices get different colors.
This allows computing multiple rows in a single AD pass.
```

## MkDocs Conventions

### Autodoc Directives

Reference pages use mkdocstrings autodoc syntax:

```markdown
::: asdex.jacobian
```

This pulls the docstring from the source code.
Keep docstrings in Google style.

### Shared Docstring Fragments

Public docstrings use `{placeholder}` tokens
(argument descriptions shared across the API, see `src/asdex/_docstrings.py`).
The runtime `@_fill_doc` decorator interpolates them when `asdex` is imported,
but mkdocstrings reads docstrings *statically* off the AST and never imports the code,
so it would render the raw tokens.
`_griffe_extensions.py` (registered under the mkdocstrings handler in `mkdocs.yml`)
re-runs the same substitution at build time so the docs render fully.
A `--strict` build fails loudly if the extension cannot load,
and `tests/test_docstrings.py` asserts no token survives the static load.

### Admonitions

Use admonitions for callouts:

```markdown
!!! tip "Title"

    Content here.

!!! warning

    Content here.
```

### Executable Code Blocks

Use `markdown-exec` to run Python code during build and show output.
Add `exec="true"` to a fenced code block:

````markdown
```python exec="true" source="above"
print("Hello from asdex!")
````

The code runs at build time and its stdout replaces the block in the rendered page.
To show both the source code and the output, add `source="above"` or `source="below"`:

````markdown
```python exec="true" source="above"
from asdex import jacobian_coloring

import jax.numpy as jnp

x_sample = jnp.zeros(50)  # sample input for sparsity pattern detection
coloring = jacobian_coloring(lambda x: (x[1:] - x[:-1]) ** 2, x_sample)
print(coloring)
```
```
````

Use this for tutorials and how-to guides
where showing real output is more convincing than hardcoded comments.
Avoid it in explanation pages where the focus is on concepts, not code.

To render the output as its own fenced code block,
print the content wrapped in triple backticks:

```python exec="true"
print(f"```\n{coloring}\n```")
```

Keep printed output small and deterministic (types, shapes, nnz counts),
so the build stays fast and the rendered output never drifts.

### Math

Use MathJax for LaTeX:

- Inline: `\(f: \mathbb{R}^n \to \mathbb{R}^m\)`
- Display: `\[J \in \mathbb{R}^{m \times n}\]`

## Local Preview

When making major changes to the docs
(adding pages, restructuring nav, changing MkDocs config),
serve the site locally and verify the result before finishing:

```bash
uv run --group docs mkdocs serve -f docs/mkdocs.yml
```

This starts a live-reloading server at `http://127.0.0.1:8000`.
The `docs` dependency group provides mkdocs and its plugins,
and `-f docs/mkdocs.yml` points at the config (the repo has no top-level `mkdocs.yml`).
Use `uv run --group docs mkdocs build --strict -f docs/mkdocs.yml`
to catch broken links, bad anchors, and warnings.

Live reload is unreliable —
always stop and restart `mkdocs serve` to see changes.

## Navigation Structure

The nav in `mkdocs.yml` maps to Diataxis categories:

- **Home** tab → landing page (index)
- **Tutorials** tab → learning-oriented walkthroughs (getting-started)
- **How-To Guides** tab → task-oriented guides
- **Explanation** tab → concept explanations
- **Reference** tab → auto-generated API docs
- **Benchmarks** → external link to benchmark dashboard

### How-To Guide Structure

The two main how-to guides, `how-to/jacobians.md` and `how-to/hessians.md`,
mirror each other section-for-section under two tiers, `## Basics` and `## Advanced`.
Keep them in sync: a change to one usually has a counterpart in the other.
The only intentional divergences are mode selection
(Jacobians choose row vs column coloring;
Hessians use symmetric coloring plus an HVP-mode choice)
and PyTree *outputs*, which appear only in the Jacobian guide
since a Hessian requires a scalar output.
The *Skipping Decompression* Advanced section is the user-facing surface of the `decompression` package's compressed API:
it covers the `compressed_*` / `value_and_compressed_*` entry points that stop at the compressed matrix \(B\)
and the `decompress` / `decompress_data` functions that recover the sparse matrix.
It is mirrored in both guides;
its reference counterparts are the *Compressed Jacobian* / *Compressed Hessian* and *Decompression* groups
in `reference/jacobian.md`, `reference/hessian.md`, and `reference/index.md`.
Correctness checking lives in its own task page, `how-to/verification.md`,
which both guides link to from a short *Verifying Results* pointer.

The **Features** lists in `README.md` and `docs/index.md` are kept in lock-step.
They hold the same entries in the same order,
differing only in link style:
the README uses absolute `https://adrianhill.de/asdex/...` URLs (it also renders on GitHub and PyPI),
while `index.md` uses relative `.md` links.
A change to one list should be mirrored in the other.
