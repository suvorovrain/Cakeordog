#!/bin/sh -e
echo "Format code"
echo "Running black"
uv run black src/ tests/ docs/
echo "Running ruff --fix"
uv run ruff check --fix src/ tests/ docs/
echo "OK"
