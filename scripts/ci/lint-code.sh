#!/bin/sh -e
echo "Lint the code"
echo "Running ruff"
uv run ruff check src/ tests/
echo "Running black --check"
uv run black --check src/ tests/
echo "Running mypy"
uv run mypy src/
echo "OK"
