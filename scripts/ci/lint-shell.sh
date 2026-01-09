#!/bin/sh -e
echo "Running shellcheck"
find scripts/ -type f \( -name "*.sh" \) -print0 | xargs -r -0 shellcheck
echo "OK"
echo "Running shfmt check"
find scripts/ -type f \( -name "*.sh" \) -print0 | xargs -r -0 shfmt -d
echo "OK"
