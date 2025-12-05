#!/bin/sh -e
echo "Running tests with coverage display"
uv run pytest tests/ -v --cov=src/cakeordog --cov-report=term-missing --cov-report=html
echo "OK"
