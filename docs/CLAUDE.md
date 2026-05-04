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

### Math

Use MathJax for LaTeX:

- Inline: `\(f: \mathbb{R}^n \to \mathbb{R}^m\)`
- Display: `\[J \in \mathbb{R}^{m \times n}\]`

## Local Preview

When making major changes to the docs
(adding pages, restructuring nav, changing MkDocs config),
serve the site locally and verify the result before finishing:

```bash
uv run mkdocs serve
```

This starts a live-reloading server at `http://127.0.0.1:8000`.
Use `uv run mkdocs build --strict` to catch broken links and warnings.

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
