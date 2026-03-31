"""Generate ground-truth sparsity fixtures from CUTEst problems via pycutest.

Extracts structural sparsity patterns using pycutest's sparse Hessian
and Jacobian routines, which call CUTEst's compiled Fortran directly.

Requires pycutest with CUTEst/SIFDecode system libraries installed.
See: https://jfowkes.github.io/pycutest/

Usage:
    uv run --with pycutest python tests/setup/generate_cutest_fixtures.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pycutest  # ty: ignore[unresolved-import]

from asdex import SparsityPattern

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MAX_SIZE = 2000

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "cutest_fixtures"


def pattern_from_sparse(mat) -> SparsityPattern:
    """Extract sparsity pattern from a scipy sparse matrix.

    Uses the stored structure (row/col indices),
    which reflects CUTEst's structural sparsity.
    """
    coo = mat.tocoo()
    return SparsityPattern.from_coo(rows=coo.row, cols=coo.col, shape=coo.shape)


def generate_hessian_fixtures(fixture_dir: Path) -> tuple[int, int]:
    """Generate Hessian fixtures for unconstrained problems."""
    out_dir = fixture_dir / "hessian"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(pycutest.find_problems(constraints="unconstrained"))
    log.info("Found %d unconstrained problems", len(names))

    generated = 0
    skipped = 0

    for name in names:
        try:
            problem = pycutest.import_problem(name)
        except Exception as exc:
            log.warning("  SKIP %s: import failed: %s", name, exc)
            skipped += 1
            continue

        if problem.n > MAX_SIZE:
            log.debug("  SKIP %s: too large (n=%d)", name, problem.n)
            skipped += 1
            continue

        try:
            H = problem.sphess(problem.x0)
            # CUTEst returns upper triangle — symmetrize via dense.
            H_dense = H.toarray()
            H_sym = H_dense + H_dense.T - np.diag(np.diag(H_dense))
            pattern = SparsityPattern.from_dense((H_sym != 0).astype(np.int8))
            pattern.save(out_dir / f"{name}.npz")
            generated += 1
            log.info(
                "  hessian/%s: %s, nnz=%d",
                name,
                pattern.shape,
                pattern.nnz,
            )
        except Exception as exc:
            log.warning("  SKIP hessian/%s: %s", name, exc)
            skipped += 1

    return generated, skipped


def generate_jacobian_fixtures(fixture_dir: Path) -> tuple[int, int]:
    """Generate Jacobian fixtures for constrained problem constraints."""
    out_dir_eq = fixture_dir / "jacobian_eq"
    out_dir_ineq = fixture_dir / "jacobian_ineq"
    out_dir_eq.mkdir(parents=True, exist_ok=True)
    out_dir_ineq.mkdir(parents=True, exist_ok=True)

    constrained: set[str] = set()
    for ctype in ("linear", "quadratic", "other"):
        constrained.update(pycutest.find_problems(constraints=ctype))
    names = sorted(constrained)
    log.info("Found %d constrained problems", len(names))

    generated = 0
    skipped = 0

    for name in names:
        try:
            problem = pycutest.import_problem(name)
        except Exception as exc:
            log.warning("  SKIP %s: import failed: %s", name, exc)
            skipped += 1
            continue

        if problem.n > MAX_SIZE or problem.m == 0:
            skipped += 1
            continue

        try:
            _c, J = problem.scons(problem.x0, gradient=True)
            J_csr = J.tocsr()
            eq_mask = problem.is_eq_cons
            ineq_mask = ~eq_mask

            if eq_mask.any():
                J_eq = J_csr[eq_mask]
                pattern = pattern_from_sparse(J_eq)
                pattern.save(out_dir_eq / f"{name}.npz")
                generated += 1
                log.info(
                    "  jacobian_eq/%s: %s, nnz=%d",
                    name,
                    pattern.shape,
                    pattern.nnz,
                )

            if ineq_mask.any():
                J_ineq = J_csr[ineq_mask]
                pattern = pattern_from_sparse(J_ineq)
                pattern.save(out_dir_ineq / f"{name}.npz")
                generated += 1
                log.info(
                    "  jacobian_ineq/%s: %s, nnz=%d",
                    name,
                    pattern.shape,
                    pattern.nnz,
                )
        except Exception as exc:
            log.warning("  SKIP jacobian/%s: %s", name, exc)
            skipped += 1

    return generated, skipped


def main() -> int:
    """Generate all CUTEst fixtures."""
    log.info("Fixture directory: %s", FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("\nGenerating Hessian fixtures from unconstrained problems...")
    h_gen, h_skip = generate_hessian_fixtures(FIXTURE_DIR)

    log.info("\nGenerating Jacobian fixtures from constrained problems...")
    j_gen, j_skip = generate_jacobian_fixtures(FIXTURE_DIR)

    total_gen = h_gen + j_gen
    total_skip = h_skip + j_skip
    log.info("\nDone: %d fixtures generated, %d skipped", total_gen, total_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
