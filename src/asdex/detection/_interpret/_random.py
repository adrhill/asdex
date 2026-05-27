"""Propagation rules for random primitives.

Random number generation is not differentiable,
so all random primitives produce outputs with empty dependency sets.
"""

from jax._src.core import JaxprEqn

from ._commons import StateIndices, atom_numel, empty_index_sets


def prop_random(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Random primitives have zero derivative with respect to inputs.

    Random number generation is not differentiable,
    so all output dependency sets are empty regardless of inputs.

    This covers:
    - `random_seed`: generates a PRNG key from a seed value
    - `random_unwrap`: extracts raw bits from a typed key
    - `random_wrap`: wraps raw bits into a typed key
    - `random_split`: splits a key into multiple subkeys
    - `random_fold_in`: derives a new key by folding in data
    - `random_bits`: generates random bits from a key

    Example: key = PRNGKey(0), noise = normal(key, (3,))
        Input state_indices:  [{0}, {1}, {2}]  (for some traced input x)
        Output state_indices: [{}, {}, {}]     (empty, no dependence on x)

    Jaxpr:
        invars: varies by primitive (seed literal, key, shape, etc.)
        outvars: random key or random values
    """
    for outvar in eqn.outvars:
        state_indices[outvar] = empty_index_sets(atom_numel(outvar))
