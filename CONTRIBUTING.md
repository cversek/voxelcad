# Contributing to VoxelCAD

## Setup

```bash
git clone <repo-url>
cd voxelcad
make install    # editable install + dev deps
make build      # compile Cython extensions
make test       # verify everything works
```

## Development Workflow

1. Create a branch from `main`
2. Make changes
3. Run `make test` — all tests must pass
4. If you changed rendering logic, run `make benchmark` to check for regressions
5. If you added/changed doc examples, run `make docs-images` to regenerate PNGs
6. Submit a pull request

## Code Style

- Follow existing patterns in the codebase
- Instrument render methods with `TIMING_START`/`TIMING_END` from `voxelcad.debug`
- Use packed boolean storage (`np.packbits(..., order='F', bitorder='big')`) for voxel data
- Return dicts (not tuples) from functions with multiple return values

## Adding a Primitive

See [Extension Guide](docs/developer/extension-guide.md) for the full walkthrough. Summary:

1. Subclass `VoxelModel`, set `self.grid` in `__init__`
2. Implement `_render_numpy()` (required) and `_render_cython()` (optional)
3. Handle `M4inv` parameter for transform support
4. Export from `__init__.py`
5. Add tests and a doc example in `docs/user/geometry-catalog.md`

## Tests

```bash
make test                          # full suite
python -m pytest tests/ -k "name"  # filter
```

Tests use 32^3 grids for speed. Shared fixtures are in `tests/conftest.py`. See [Testing Strategy](docs/developer/testing-strategy.md).

## Documentation

User docs are in `docs/user/`, developer docs in `docs/developer/`.

Code examples in markdown docs are automatically extracted and rendered to PNGs by `docs/tools/extract_examples.py`. After editing docs:

```bash
make docs-images
```

Images go to `docs/user/_images/{doc_stem}/{heading_slug}_{idx}_{var}.png` and are referenced with relative paths in the markdown.

## Reporting Issues

Open a GitHub issue with:

- What you expected
- What happened instead
- Steps to reproduce
- Python version, OS, and whether Cython extensions are compiled (`python -c "import voxelcad; print(voxelcad.ENV.use_cython)"`)
