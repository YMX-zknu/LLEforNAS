# Contributing

Create a focused branch, add tests for behavioral changes, and run:

```bash
ruff check .
pytest
python -m compileall -q src tests
```

Keep the public tensor contract batch-first: `[B, T, C, H, W]`. New datasets and search spaces must be registered through their registries rather than added to command-specific conditionals.
