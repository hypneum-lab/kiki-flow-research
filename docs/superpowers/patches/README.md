# micro-kiki integration patches

Ready-to-apply patches for the three remaining integration dependencies
documented in `../integration-notes.md`.

| Patch | Dep | Risk | Resolves |
|---|---|---|---|
| `00-micro-kiki-dep1-ane-router-dynamic-N.patch` | 1 | low | Hard-coded 37 in ANERouter.route(). Loads N from model spec dynamically; defaults to 32. |
| `01-micro-kiki-dep4-feature-flag.patch` | 4 | low | `KIKI_FLOW_ENABLED` env var plumbing in Config dataclass. |

Dep 2 (T3 surrogate at state_dim=128) is already resolved in this repo
(`kiki_flow_core/track3_deploy/weights/v0.2-d128.safetensors`).

Dep 3 (query-string plumbing through the router call path) requires a
refactor of the MoE router's interface to pass the raw query string
alongside the hidden-state vector. This is a larger design change and
is not covered by a patch file; the two sketched options are recorded
in `../integration-notes.md` section 3.

## Applying the patches

```bash
cd ~/KIKI-Mac_tunner
git checkout -b feat/kiki-flow-integration
git apply docs/superpowers/patches/00-micro-kiki-dep1-ane-router-dynamic-N.patch
git apply docs/superpowers/patches/01-micro-kiki-dep4-feature-flag.patch
git add src/serving/ane_router.py src/config.py
git commit -m "feat(micro-kiki): integrate kiki-flow deps 1 and 4"
# run tests
uv run pytest tests/serving
# open PR
gh pr create --title "feat: integrate kiki-flow deps (ANE N, feature flag)"
```

These patches are **unified diffs against the paths observed on
Studio on 2026-04-18**. Line numbers may drift if the upstream
files have moved; rebase with `patch -p1 --fuzz=3` or apply
manually if `git apply` rejects.
