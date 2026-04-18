# tests — pytest + golden NPZ regression

Layout mirrors the source: root tests exercise `kiki_flow_core/` top-level;
`tests/track{1,2,3}_*/` mirror the matching track subpackage. Coverage
target is the published 95 % (`--cov=kiki_flow_core` is in pyproject
`addopts`, so any `pytest` invocation already measures it).

## Conventions

- `conftest.py` seeds NumPy globally per test via autouse
  (`np.random.seed(42)`). Do not depend on this for new stochastic tests —
  take an explicit seed and use `np.random.default_rng(seed)` inside the
  test body. The autouse fixture exists to protect legacy tests from
  cross-test leakage, not to bless hidden randomness.
- Markers: `slow` (tests > 1 s), `golden` (numerical regression against
  `tests/golden/*.npz`). Mark new slow or golden tests accordingly so CI
  filtering keeps working.
- `pythonpath = ["."]` in pyproject — tests import `kiki_flow_core`
  directly. Do not add `sys.path.insert(...)` shims.
- Tolerances are named constants per file (e.g. `SYMMETRY_TOL`,
  `GOLDEN_TOL`, `GAUSSIAN_DISCRETIZATION_TOL`). Keep naming intent-bearing;
  raw floats in assertions trip ruff `PLR2004`.

## Golden fixtures

`tests/golden/{advection_1d_gaussian,diffusion_gaussian,sinkhorn_5x5}.npz`
pin exact numerical outputs. When a legitimate algorithmic change moves
these numbers:

1. Regenerate the NPZ with a deterministic one-shot script (commit the
   script alongside the NPZ change).
2. Record the reason in the commit message — which function changed and
   why the new value is correct.
3. Do NOT widen the `GOLDEN_TOL` instead of regenerating. The point of a
   golden test is bit-level stability; loosening the tolerance turns it
   into a smoke test.

## Smoke / E2E tests

`test_smoke_e2e.py` wires `FlowState -> JKOStep -> assert_invariants` with
a `ZeroF` functional. Keep this test fast (< 1 s) — it's the canary for
API-level breakage across the three tracks.

## Anti-patterns (domain-specific)

- Flaky tolerance-chasing: if a test passes on seed 42 but fails on seed
  43, that's a real non-robustness, not a test bug. Debug the algorithm.
- Asserting `isclose` with `atol=1.0` on a density — you are comparing
  probabilities, 1.0 is the entire simplex.
- Running `ot.sinkhorn` in a test without `method="sinkhorn_log"`. Tests
  exercise small epsilons; the default kernel underflows.
- Importing from `kiki_flow_core.track1_perf` in a `track2_paper/` test
  (or vice versa). Shared helpers belong in root `kiki_flow_core/`.
- Depending on filesystem state (e.g. reading `paper/stats.json` in a
  test). Tests must run on a fresh clone before any script has been run.
