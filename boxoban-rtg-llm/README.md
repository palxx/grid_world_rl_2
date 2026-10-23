# boxoban-rtg-llm

Research code testing whether Qwen3-8B can be conditioned on a requested return-to-go (RTG) in DeepMind Boxoban.
Read `docs/DECISIONS.md` first: it separates approved decisions, verified facts, estimates and open questions.

## Status
Phase 1 (environment) only. Implemented and tested: official-level loader, environment adapter checked against gym-sokoban 0.0.6, reward semantics, CLI (`hardware-check`, `download-levels`, `inspect-level`, `replay`, `smoke-test`). Not implemented: solver, trajectories, language serialisation, training, evaluation, plots.

## Quick start
    pip install -e ".[dev]"
    boxoban-rtg download-levels          # or set BOXOBAN_LEVELS_DIR
    pytest -q
    boxoban-rtg smoke-test
    boxoban-rtg hardware-check           # on the GPU machine
