---
name: add-handler
description: Add a precise primitive handler to the jaxpr interpreter, replacing a conservative fallback.
argument-hint: "[primitive-name]"
disable-model-invocation: true
---

# Add precise handler for `$ARGUMENTS`

Add a precise propagation handler for the `$ARGUMENTS` primitive,
replacing the conservative fallback.

## Workflow

### 1. Research

Do all research in this step using extended **planning mode**.

Before writing code:
- Read the JAX docs for the primitive: fetch `https://docs.jax.dev/en/latest/_autosummary/jax.lax.$ARGUMENTS.html`
- Read `src/asdex/detection/_interpret/CLAUDE.md` for conventions (docstring style, semantic line breaks, handler structure)
- Read `src/asdex/detection/_interpret/_commons.py` to understand available utilities
- Read an existing handler with similar structure (e.g. `_pad.py`, `_transpose.py`, `_reduction.py`) as a reference
- Read the existing test in `tests/_interpret/test_internals.py` for the primitive (search for `$ARGUMENTS`)
- Read `src/asdex/detection/_interpret/__init__.py` to see the current dispatch and fallback setup

Understand the primitive's semantics:
how do input and output element indices map to each other?
What is the Jacobian structure (permutation, selection, block-diagonal, etc.)?

### 2. Implement handler

- Create `src/asdex/detection/_interpret/_$ARGUMENTS.py` with `prop_$ARGUMENTS(eqn, deps)`.
- Follow the handler docstring style from `_interpret/CLAUDE.md`.
- If the primitive preserves or predictably transforms input values,
  propagate `const_vals` so downstream handlers can stay precise.
- Handle zero-sized arrays: if the output has zero elements,
  return `[]` early before any coordinate-mapping logic
  that would crash on zero-sized shapes.

### 3. Wire up dispatch

In `src/asdex/detection/_interpret/__init__.py`:
- Import the new handler
- Add a `case "$ARGUMENTS":` branch in `prop_dispatch` calling the handler
- Remove `"$ARGUMENTS"` from the conservative fallback `case` group

### 4. Update tests

In `tests/_interpret/test_internals.py`:
- Update the existing test: change expected values from dense (`np.ones`) to the precise pattern
- Remove the `@pytest.mark.fallback` marker and `TODO` comments

Create `tests/_interpret/test_$ARGUMENTS.py` with thorough tests:
- Multiple dimensionalities (1D, 2D, 3D, 4D where applicable)
- Broadcasting shapes: size-1 dimensions that broadcast (e.g. `(3,4)` op `(3,1)`)
- Non-square shapes (e.g. `(3,4)` not `(4,4)`) so dimension mix-ups are caught
- Edge cases (size-0 dimensions, identity/trivial parameters)
- Real-world usage patterns (e.g. `jnp` functions that lower to this primitive)
- For at least one test per dimensionality, verify precision by comparing the detected
  pattern against `(np.abs(jax.jacobian(f)(x)) > 1e-10)` using `assert_array_equal`.
  Choose test functions that avoid local sparsity (e.g. multiply by zero) so the
  numerical Jacobian matches the structural pattern.

### 5. Verify

Run in order:
```bash
uv run ruff check src/asdex/detection/_interpret/_$ARGUMENTS.py
uv run pytest tests/_interpret/test_$ARGUMENTS.py -v
uv run pytest tests/_interpret/test_internals.py -v
uv run pytest tests/ -x
```

### 6. Adversarial tests

Reread the JAX docs for the primitive: fetch `https://docs.jax.dev/en/latest/_autosummary/jax.lax.$ARGUMENTS.html`

Try to break the implementation by testing inputs the handler might not expect:

- **Dimensionality**: 1D, 2D, 3D, and higher — if any are missing, add them.
- **Asymmetric shapes**: always use non-square shapes (e.g. `(3,4)` not `(4,4)`) so that dimension transposition bugs are caught.
- **Degenerate shapes**: size-0 dimensions (must not crash, should return empty index sets), size-1 dimensions, scalar inputs (where the primitive supports them).
- **Boundary parameters**: empty parameter lists, all-dimensions, single-dimension, negative indices (if applicable).
- **Compositions**: the primitive chained with itself (e.g. double-reverse, transpose-of-transpose) or with related ops.
- **Non-contiguous patterns**: inputs where dependencies are not simply `{i}` per element (e.g. from a prior broadcast or reduction) to verify `.copy()` and set merging behave correctly.
- **Conservative audit**: for each test case, verify the result is strictly sparser than what `conservative_indices()` would produce. If the handler silently falls back on any shape the primitive supports, investigate.
  If the fallback cannot be fixed immediately, you **must** add a `@pytest.mark.fallback` test with a `TODO(primitive)` comment showing the precise expected pattern.
  Catching conservative patterns is extremely valuable for future development.
- **Const chain**: if the primitive can appear between a literal and a downstream consumer (e.g. type conversions, reshapes, broadcasts on index arrays), write a test composing it with a downstream gather and verify the gather resolves precisely.

For each new test, verify the expected output by hand or against `jax.jacobian`.
Update and re-verify the handler if any test reveals a bug.

### 7. Simplify

Review the implementation with fresh eyes and look for opportunities to reduce complexity:

- **Vectorize loops**: can per-element Python loops be replaced with numpy operations?
  Pattern: build a flat permutation or index array with `np.arange`, `np.flip`, `np.transpose`, `np.indices`, or `np.ravel_multi_index`,
  then index into `in_indices` in a single list comprehension.
  See `_rev.py`, `_reshape.py`, `_concatenate.py`, and `_broadcast.py` for examples.
- **Remove unused imports**: after vectorizing, utilities like `flat_to_coords`, `row_strides`, and `numel` may no longer be needed.
- **Eliminate intermediate variables**: if a value is computed and used only once, inline it.
- **Simplify special cases**: can a special-case branch be absorbed into the general case?

After any change, re-run verification (step 6).

### 8. Update docs

- `TODO.md`: check off the primitive and its test items
- `src/asdex/detection/_interpret/CLAUDE.md`: add the new module to the file listing
