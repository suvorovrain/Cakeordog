#!/bin/sh -e
echo "Format code"
echo "Running black"
uv run black src/ tests/
echo "Running ruff --fix"
uv run ruff check --fix src/ tests/
echo "OK"
