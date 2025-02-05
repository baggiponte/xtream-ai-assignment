set shell := ["zsh", "-uc"]
set positional-arguments

# List all available recipes
help:
  @just --list

# Lock dependencies
lock:
  @{{just_executable()}} needs pdm

  pdm lock --dev --group=:all

# Create a git repo if not exists, install dependencies and pre-commit hooks
install:
  @{{just_executable()}} needs pdm

  pdm install --dev --group=:all
  pdm run pre-commit install --install-hooks
  pdm run nbstripout --install

# Run the training pipeline
@train:
  pdm run pythonm scripts/train.py

# Update dependencies and update pre-commit hooks
update: lock
  pdm update
  pdm run pre-commit install-hooks
  pdm run pre-commit autoupdate

# Launch a jupyter lab instance
@lab:
  pdm run jupyter-lab

# Format code with black and isort
@fmt:
  pdm run black -- src tests
  pdm run blacken-docs -- src/**/*.py tests/*.py
  pdm run ruff --select=I001 --fix -- src tests

# Lint the project with Ruff
@lint:
  pdm run ruff -- src tests

# Perform static type checking with mypy
@typecheck:
  pdm run mypy -- src tests

# Audit dependencies with pip-audit
@audit:
  pdm run pip-audit
  pdm run deptry -- src

# Assert a command is available
[private]
needs *commands:
  #!/usr/bin/env zsh
  set -euo pipefail
  for cmd in "$@"; do
    if ! command -v $cmd &> /dev/null; then
      echo "$cmd binary not found. Did you forget to install it?"
      exit 1
    fi
  done
