# Reward semantics

Reference: gym-sokoban 0.0.6 `SokobanEnv._calc_reward`, evaluated once per step **after** the move.

| Event | Reward | Units of 0.1 |
|---|---|---|
| Every step (including blocked / no-op moves) | -0.1 | -1 |
| Boxes-on-goal count increased vs. previous step | +1 | +10 |
| Boxes-on-goal count decreased | -1 | -10 |
| All goals covered by boxes | +10 | +100 |

Combinations: push onto a goal 0.9; push off a goal -1.1; the final push 10.9; blocked step -0.1.
The +1 / -1 test compares counts, so it applies once per step regardless of how many boxes changed (at most one box moves per step).

- Rewards are computed from the executed environment transition only, as integer units; `reward = units / 10`.
- Episode return = sum of step rewards. RTG(t) = sum of rewards from t to T-1; RTG(T) = 0. Closed loop: `remaining(t+1) = remaining(t) - reward(t)`.
- Solved = every goal holds a box (terminated). Truncated = step budget exhausted and not solved (default 120).
- No reward shaping is used in the primary experiment.
