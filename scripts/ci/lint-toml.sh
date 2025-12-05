#!/bin/sh -e
echo "Lint the pyproject.toml"
echo "Run taplo lint"
taplo lint pyproject.toml
echo "Run taplo format --check"
taplo format --check pyproject.toml
echo "OK"
