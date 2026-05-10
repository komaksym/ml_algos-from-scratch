# ML Algorithms From Scratch

Small Python workspace for implementing classic ML and PyTorch interview exercises from scratch: training loops, simple neural nets, optimization basics, attention blocks, and similar junior ML engineer practice.

## Setup

```bash
uv sync --group dev
uv run prek install --overwrite
```

`prek` is the installed Git hook runner because it is fast. The same hook config also works with `pre-commit`.

## Checks

```bash
uv run prek run --all-files
uv run pre-commit run --all-files
```

Direct Ruff checks:

```bash
uv run ruff format --check .
uv run ruff check .
```

## Layout

- `in_numpy/`: NumPy-first implementations for fundamentals.
- `in_torch/`: PyTorch implementations and interview-style training loops.
