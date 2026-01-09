#!/bin/sh -e
echo "Lint the code"
echo "Running ruff"
uv run ruff check src/ tests/ docs/
echo "Running black --check"
uv run black --check src/ tests/ docs/
echo "Running mypy"
uv run mypy src/ docs/
echo "Running ruff on notebook"
uv run nbqa ruff notebook/
echo "Running black --check on notebook"
uv run nbqa black --check notebook/
echo "Running mypy on notebook"
uv run nbqa mypy notebook/
echo "OK"
