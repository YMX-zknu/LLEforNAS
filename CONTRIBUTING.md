# Contributing

Create a focused branch, add tests for behavioral changes, and run:

```bash
ruff check .
pytest
python -m compileall -q src tests
```

Keep the public tensor contract batch-first: `[B, T, C, H, W]`. Models passed to `estimate_ftle` must implement `step(frame, state)` and return their recurrent state as a tuple of tensors. Check proxy-score semantics against the paper before modifying the search code.
