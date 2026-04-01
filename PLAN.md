# Plan

Delete items when completed.

Effort and priority in brackets after each heading:
- Effort: `S` (< 1 session), `M` (1-2 sessions), `L` (3+ sessions)
- Priority: `!` high, `!!` urgent (e.g. `[M!]`, `[S!!]`)

---

## Now

### Add inline comments to expected matrices in test files [S]

Annotate expected sparsity matrices with inline comments
that explain what each row computes, directly on the array literal.
This makes test expectations self-documenting
without requiring the reader to cross-reference the docstring.

Move explanations out of docstrings and into the matrix itself.
Keep docstrings short (one-line summary + setup context).

**Style:**

```python
expected = np.array(
    [
        [1, 1, 1],  # carry_out = x[0] + x[1] + x[2]
        [0, 0, 0],  # ys[0] = carry_init = 0
        [1, 0, 0],  # ys[1] = x[0]
        [1, 1, 0],  # ys[2] = x[0] + x[1]
    ],
    dtype=int,
)
```

When consecutive rows share the same pattern (e.g. elementwise over a 2-element array),
annotate only the first row of each group.

Apply to all `tests/_interpret/test_*.py` files, not just `test_scan.py`.

## Later

### Symbolic index tracking for gather/scatter composition [L]

Composing gather → compute → scatter-add through a shared accumulator
unions deps from different members,
making `output[*]` appear to depend on `input[*]`.
Fixing this requires tracking that `gather` at `[2,3]` then `scatter-add` at `[5]`
only creates `output[5] ← input[2,3]`.

Architecturally difficult: the current framework evaluates each primitive independently.

**CUTEst impact** (see `CUTEST_ANALYSIS.md`):
TRO3X3/4X4/5X5/6X2/11X3 (298–8,554 extra nnz),
HAIFAM (12,039 extra nnz), HAIFAS (13 extra nnz).

### Second-order analysis for `reduce_sum` Hessian precision [L]

The VJP of `reduce_sum` broadcasts cotangents back,
so `J^T J` appears fully dense even when `J` columns don't all overlap.
For least-squares `f(x) = sum(r_i(x)^2)`,
the Hessian has nonzeros only where `J` columns overlap,
but `sum` unions all residual dependencies.
Fixing this requires tracking which *pairs* of inputs interact,
beyond the current first-order index-set framework.

**CUTEst impact** (see `CUTEST_ANALYSIS.md`):
ARGLINA (39,800 extra nnz), VANDANMSLS (468),
plus 14 moderately conservative and 5 near-exact Hessian problems.

## Ideas

- Algebraic cancellation detection
  (e.g. `f(x) - f(x) = 0` — requires symbolic simplification beyond index-set propagation).
  Would fix TENBARS1–4 (62 extra nnz each), FLETCHER (4), S316-322 (2).
- Transfer matrix optimization for `scan` (see below).

### Transfer matrix optimization for `scan` [M]

The current scan handler calls `prop_jaxpr` once per timestep: `O(length × prop_cost)`.
For large `length` this may become a bottleneck.

#### Approach

The body jaxpr is a fixed function `(carry, x_slice) → (new_carry, y_slice)`.
Its dependency structure can be decomposed into four boolean transfer matrices:

- **C→C**`[i,j]`: `carry_out[i]` depends on `carry_in[j]`
- **X→C**`[i,j]`: `carry_out[i]` depends on `x_slice[j]`
- **C→Y**`[i,j]`: `y_slice[i]` depends on `carry_in[j]`
- **X→Y**`[i,j]`: `y_slice[i]` depends on `x_slice[j]`

These can be extracted with a single `prop_jaxpr` call
using indicator index sets (`carry_in[i] = {i}`, `x_slice[j] = {carry_size + j}`),
then reading off which indicators appear in each output.

Forward simulation becomes boolean matrix-vector operations:

```
carry_deps[0] = init_carry_deps
for t in range(length):          # reversed(range(length)) if reverse=True
    xs_t = xs_deps[t]
    ys_deps[t][i] = ∪{carry_deps[j] : C→Y[i,j]} | ∪{xs_t[j] : X→Y[i,j]}
    carry_deps[i] = ∪{carry_deps[j] : C→C[i,j]} | ∪{xs_t[j] : X→C[i,j]}
```

Cost: `O(prop_cost + length × (carry_size + xs_slice_size)²)`.
The `prop_jaxpr` call happens once; the forward loop is cheap set unions.

#### Soundness constraint

Transfer matrices are extracted once and reused for all timesteps,
so they must be conservative for *every* possible carry value, not just `carry_init`.

**Const values for carry and x_slice must NOT be passed** during extraction.
Only closed-over body constants (which don't change across timesteps) may use const info.
Otherwise, const-killing produces matrices that are too tight:
e.g. body `(c, x) → (c + x, c * x)` with `carry_init = 0`
would kill X→Y (since `0 * x = 0`),
but at `t=1` carry is non-zero and `y = carry * x` genuinely depends on `x`.

#### Precision trade-off vs current approach

The current per-timestep approach is more precise at `t=0`
when `carry_init` is a known constant,
because `prop_jaxpr` naturally handles const-killing.
Transfer matrices lose this — they overapproximate at `t=0`
for bodies like `y = carry * x` where `carry_init = 0`.
This is conservative (safe), just not tight.

## Cross-task insights

- **`state_consts` sharing across `prop_jaxpr` calls is safe**
  when the body jaxpr is the same object each call
  and carry/xs vars are never added to `state_consts`.
  All body-internal const propagation is idempotent.
- **`atom_shape` in `_commons.py`** exists and should be used
  instead of raw `tuple(getattr(var.aval, "shape", ()))` in handlers.
