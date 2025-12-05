#!/bin/sh -e
echo "Format pyproject.toml"
echo "Running taplo format"
taplo format pyproject.toml
echo "OK"
