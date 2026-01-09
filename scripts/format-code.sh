#!/bin/sh -e
echo "Format code"
echo "Running black"
uv run black src/ tests/ docs/ notebook/
echo "Running ruff --fix"
uv run ruff check --fix src/ tests/ docs/ notebook/
echo "Running black on notebook"
uv run nbqa black notebook/
echo "Running ruff --fix"
uv run nbqa ruff --fix notebook/
echo "OK"
