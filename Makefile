# VoxelCAD Build Makefile — run from repo root
#
# Usage:
#   make              # Pull pa_manny/main, pip install, build Cython extensions
#   make BRANCH=main  # Pull a different branch

BRANCH ?= pa_manny/main
REMOTE ?= origin

.PHONY: all pull install build_ext

all: pull install build_ext

pull:
	@CURRENT=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$CURRENT" != "$(BRANCH)" ]; then \
		git fetch $(REMOTE) && \
		git checkout $(BRANCH) 2>/dev/null || git checkout -b $(BRANCH) $(REMOTE)/$(BRANCH); \
	fi
	git pull $(REMOTE) $(BRANCH)

install: pull
	pip install -e .

build_ext: install
	python setup.py build_ext --inplace
	@echo "--- Build complete: $$(git log --oneline -1) ---"
