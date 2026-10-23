#!/usr/bin/env bash
# Run inside WSL2 Ubuntu. Does NOT install any NVIDIA display driver (WSL2 uses the Windows driver).
set -euo pipefail
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,data]"
# GPU stack: versions are pinned and recorded in docs/REPRODUCIBILITY.md after the first successful hardware-check.
# pip install torch ...   (added in the training phase)
boxoban-rtg download-levels
boxoban-rtg hardware-check
pytest -q
boxoban-rtg smoke-test
