# Tests

Run the suite from the repository root:

```bash
pytest -q
```

The suite focuses on deterministic project logic in `src/`:

- acquisition functions
- data loading / updates / historical records
- candidate generation and exhaustive search ordering
- GP surrogate helpers
- benchmark progression summaries
- quantitative validation helpers
- one lightweight plotting helper smoke test

Three regression tests are intentionally marked `xfail(strict=True)` because the current source has known inconsistencies:

1. `search_region(..., upper_radius=...)` ignores `upper_radius`.
2. `exhaustive_kappa_sensitivity_4d(..., slabs_per_batch=...)` is incompatible with the final `_original_grid_blocks_4d` wrapper.
3. `plot_best_observed_progression_2d` is annotated to return `Axes` but currently returns `None`.

Once each source issue is fixed, the corresponding `xfail` marker should be removed.
