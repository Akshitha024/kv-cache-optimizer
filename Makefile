.PHONY: install lint fmt typecheck test bench report pdf clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -v

bench:
	uv run kvopt bench --out-dir runs/latest --n-requests 200 --seed 17

report:
	uv run kvopt report --out-dir runs/latest

pdf:
	cd docs/_report && pandoc research_report.md \
	    -o ../research_report.pdf \
	    --pdf-engine=xelatex --toc --toc-depth=2 --number-sections \
	    -V geometry:margin=1in -V fontsize=11pt \
	    -V mainfont="Helvetica" -V monofont="Menlo" \
	    -V linkcolor=blue -V urlcolor=blue -V linestretch=1.15 \
	    || echo "pandoc + xelatex required to render docs/research_report.pdf"

clean:
	rm -rf runs/* .pytest_cache .mypy_cache .ruff_cache
