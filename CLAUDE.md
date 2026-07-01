# asdex - Automatic Sparse Differentiation in JAX

This package implements [Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) (ASD) in JAX.

## Overview

ASD exploits sparsity to reduce the cost of computing sparse Jacobians and Hessians:

1. **Detection**: Analyze the jaxpr computation graph to detect the global sparsity pattern
2. **Coloring**: Assign colors to rows so that rows sharing non-zero columns get different colors
3. **Decompression**: Compute one VJP/HVP per color instead of one per row, then extract the sparse matrix

## Structure

```
src/asdex/
├── __init__.py         # Public API
├── _arguments.py       # Argument handling and validation (argnums, kwargs binding, dtype/structure checks)
├── _check.py           # Correctness checks (check_jacobian_correctness, check_hessian_correctness)
├── _display.py         # Display/formatting utilities
├── _errors.py          # Errors and warnings (VerificationError, InvalidColoringError, DenseColoringWarning)
├── _plotting.py        # Matplotlib visualizations for SparsityPattern and ColoredPattern
├── coloring/           # Graph coloring (row, column, symmetric) and convenience functions
├── decompression/      # Compress (one VJP/JVP/HVP per color, producing B) then decompress it into the sparse matrix, plus the public API
├── detection/          # Jacobian and Hessian sparsity detection via jaxpr analysis
│   └── _interpret/     # Custom jaxpr interpreter for index set propagation
├── _differentiation.py # Batched-AD engine: one VJP/JVP/HVP per color, producing the compressed matrix B
├── _pattern.py         # SparsityPattern and ColoredPattern data structures
├── _pytree.py          # Generic PyTree <-> flat array plumbing (flatten, unflatten, size, dtype)
└── _types.py           # Type aliases for AD modes and output formats (JacobianMode, HessianMode, OutputFormat) and mode validators
```

The interpreter internals are described in `src/asdex/detection/_interpret/CLAUDE.md`.
The decompression internals are described in `src/asdex/decompression/CLAUDE.md`.
The structure of the test folder is described in `tests/CLAUDE.md`.

## Development

```bash
uv run ruff check --fix .  # lint + auto-fix
uv run ruff format .       # format
uv run ty check            # type check
uv run pytest              # run tests
```

## Code style

- Favor `match` statements over long if-else chains.
  Use explicit cases and default to `case _ as unreachable: assert_never(unreachable)`.
- Use plain `# Section name` comments for section separators,
  not banner-style `# -- Section name ---`.

## Architecture

### Jacobians

```
jacobian(f, input_shape)(x)          # one-call API
jacobian_from_coloring(f, coloring)(x)  # from pre-computed coloring
  │
  ├─ 1. DETECTION
  │     jacobian_sparsity(f, input_shape)
  │     ├─ make_jaxpr(f) → jaxpr
  │     ├─ _prop_jaxpr() → index sets
  │     └─ SparsityPattern
  │
  ├─ 2. COLORING
  │     jacobian_coloring_from_sparsity(sparsity)
  │
  └─ 3. DECOMPRESSION
        Compress:   one VJP or JVP per color → B
        Decompress: scatter B into the pattern

Precompute: jacobian_coloring(f, shape) = detect + color
```

### Hessians

```
hessian(f, input_shape)(x)          # one-call API
hessian_from_coloring(f, coloring)(x)  # from pre-computed coloring
  │
  ├─ 1. DETECTION
  │     hessian_sparsity(f, input_shape)
  │     └─ jacobian_sparsity(grad(f), input_shape)
  │
  ├─ 2. COLORING
  │     hessian_coloring_from_sparsity(sparsity)
  │
  └─ 3. DECOMPRESSION
        Compress:   one HVP per color (fwd-over-rev) → B
        Decompress: scatter B into the pattern

Precompute: hessian_coloring(f, shape) = detect + color_symmetric
```

The decompression step splits into compress (build the dense `B`) and decompress (scatter `B` into the pattern),
which never import each other.
The `compressed_*` / `value_and_compressed_*` entry points stop at `B`,
and `decompress` / `decompress_data` turn a caller-supplied `B` back into a sparse matrix.
See `src/asdex/decompression/CLAUDE.md`.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for all commit messages (e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
For breaking changes, add `!` after the type (e.g. `feat!:`).

## Design philosophy

When writing new code, adhere to these design principles:

- **Minimize complexity**: The primary goal of software design is to minimize complexity—anything that makes a system hard to understand and modify.

- **Information hiding**: Each module should encapsulate design decisions that other modules don't need to know about, preventing information leakage across boundaries.

- **Pull complexity downward**: It's better for a module to be internally complex if it keeps the interface simple for others. Don't expose complexity to callers.

- **Favor exceptions over wrong results**: Raise errors for unknown edge cases rather than guessing.
