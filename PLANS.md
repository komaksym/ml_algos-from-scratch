# Plans

## Repo Tooling Setup

Summary: configure this repository as a small Python/uv workspace for ML algorithm interview practice, with `prek` and `pre-commit` sharing one hook config.

Milestones:

1. Configure project metadata, runtime dependencies, dev dependencies, Ruff, and file ignore rules.
2. Add compatible `prek`/`pre-commit` hooks for Ruff formatting/linting and basic file hygiene.
3. Remove notebook checkpoint files from the Git index while preserving local copies.
4. Sync dependencies, install the `prek` Git hook shim, and run validation through both hook runners.
