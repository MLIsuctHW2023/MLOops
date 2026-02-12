#!/bin/bash
set -e

echo 
ruff format .

echo 
ruff check .

echo 
mypy . --config-file mypy.ini

echo 