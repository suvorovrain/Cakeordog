#!/bin/sh -e
echo "Format shell scripts"
echo "Running shfmt"
shfmt -w scripts
echo "OK"
