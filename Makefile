# Makefile for easy development workflows.
# See docs/development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install lint test upgrade build clean results

default: install lint test 

install:
	uv sync --all-extras

lint:
	uv run python devtools/lint.py

test:
	uv run pytest

upgrade:
	uv sync --upgrade --all-extras --dev

build:
	uv build

results:
	uv run pelican test-render --resolution 128 --output-dir out/test_render
	uv run pelican optimize --target images/pelican-drawing-1.jpg \
		--steps 500 --resolution 128 --save-every 50 --output-dir out/optimize
	uv run pelican greedy-refine --target images/pelican-drawing-1.jpg \
		--resolution 128 --max-shapes 35 --initial-steps 500 \
		--settle-steps 100 --reoptimize-steps 200 --max-failures 5 \
		--output-dir out/greedy_refine
	cp images/pelican-drawing-1.jpg docs/results/00_target.jpg
	cp out/test_render/pelican_test.png docs/results/01_test_render.png
	cp out/test_render/pelican_test.svg docs/results/01_test_render.svg
	cp out/optimize/pelican_optimized.png docs/results/02_optimized.png
	cp out/optimize/pelican_optimized.svg docs/results/02_optimized.svg
	cp out/optimize/optimization.gif docs/results/02_optimization.gif
	cp out/greedy_refine/round_00_initial/optimized.png docs/results/03_greedy_initial.png
	cp out/greedy_refine/final/pelican_final.png docs/results/04_greedy_final.png
	cp out/greedy_refine/final/pelican_final.svg docs/results/04_greedy_final.svg
	cp out/greedy_refine/greedy_refinement.gif docs/results/04_greedy_refinement.gif
	cp out/greedy_refine/pipeline_stages.svg docs/results/pipeline_stages.svg

clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-find . -type d -name "__pycache__" -exec rm -rf {} +
