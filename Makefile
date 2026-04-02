# VoxelCAD Build Makefile — run from repo root
#
# Usage:
#   make              # Pull, pip install, build Cython extensions
#   make build        # Just compile Cython extensions
#   make test         # Run test suite
#   make docs-images  # Regenerate documentation images
#   make help         # Show all targets
#
#   make BRANCH=main  # Pull a different branch

BRANCH ?= pa_manny/main
REMOTE ?= origin
PYTHON ?= python

.PHONY: all pull install build help test benchmark docs-images clean

all: pull install build  ## Pull, install, and build extensions

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

pull:  ## Pull latest from remote
	@CURRENT=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$CURRENT" != "$(BRANCH)" ]; then \
		git fetch $(REMOTE) && \
		git checkout $(BRANCH) 2>/dev/null || git checkout -b $(BRANCH) $(REMOTE)/$(BRANCH); \
	fi
	git pull $(REMOTE) $(BRANCH)

install: pull  ## Editable install with dev dependencies
	pip install -e ".[dev]"

build: install  ## Compile Cython extensions (OpenMP parallel kernels)
	$(PYTHON) setup.py build_ext --inplace
	@echo "--- Build complete: $$(git log --oneline -1) ---"

test:  ## Run test suite
	$(PYTHON) -m pytest tests/ -v

benchmark:  ## Run benchmarks
	$(PYTHON) -m pytest benchmarks/ -v

docs-images:  ## Regenerate documentation example images
	$(PYTHON) docs/tools/extract_examples.py
	$(PYTHON) docs/tools/render_examples.py

clean:  ## Remove build artifacts and compiled extensions
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find src -name '*.so' -delete
	find src -name '*.c' -path '*/_kernels/*' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
