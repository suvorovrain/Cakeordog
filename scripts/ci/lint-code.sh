#!/bin/sh -e
echo "Lint the code"
echo "Running ruff"
uv run ruff check src/ tests/ docs/
echo "Running black --check"
uv run black --check src/ tests/ docs/
echo "Running mypy"
uv run mypy src/ docs/
echo "OK"
