# Decisions log

Status key: **VERIFIED** = checked against source or data in this repo's tests; **APPROVED** = decided by the project owner;
**ESTIMATE** = not measured, to be replaced by measurements; **OPEN** = needs approval before the relevant phase.

## Approved by the project owner
| # | Decision | Source |
|---|---|---|
| D1 | Code lives in `boxoban-rtg-llm/` inside `palxx/grid_world_rl_2`. | owner, session start |
| D2 | "Without pretraining" means **without vs. with Boxoban fine-tuning** of Qwen3-8B (Models A and E). No from-scratch / random-init model (old "Model F" dropped). | owner |
| D3 | The prompt includes the current step index and remaining step budget (horizon 120). Fixed per experiment, not randomised. | owner |
| D4 | First runs use a single training seed; results are labelled exploratory. | owner |
| D5 | Gym-Sokoban is the behavioural reference; a minimal compatible re-implementation is the fallback if it does not install. No rule changes. | owner |

Consequence of D3: the horizon `H` and step `t` appear in every prompt, so a model trained at H=120 is only evaluated at H=120 unless a separate generalisation test says otherwise.

## Verified facts (gym-sokoban 0.0.6 source; official levels)
- **V1 Actions.** Ids 1-4 are push up/down/left/right: push an adjacent box if it can move, else plain move. Ids 5-8 are move-only. We expose UP/DOWN/LEFT/RIGHT = gym ids 1-4. Blocked actions still cost a step and are recorded (`moved_player=False`). The "[0,3]" comment in the reference is stale.
- **V2 Rewards.** Step -0.1; +1 if boxes-on-goal increased; -1 if it decreased; +10 when all boxes are on goals. The final push is therefore +10.9 (see REWARDS.md).
- **V3 Horizon.** `max_steps=120` default; done at step 120 even if unsolved. We always pass it explicitly and report `truncated` separately.
- **V4 Levels.** Counts match the project spec: unfiltered 900,000 / 100,000 / 1,000; medium 450,000 / 50,000; hard 3,332. Every level is 10x10 with 4 boxes and 4 goals, unique within its partition, none starts with a box or the player on a goal. (Cross-partition duplication has NOT been checked yet; it must be before splits are fixed.)
- **V5 Equivalence.** `tests/test_reference_equivalence.py` runs the real gym-sokoban code in lockstep with our adapter: per-step reward, player/box positions, done, moved flags, solved vs. truncated. Covers random play on official levels and tiny levels that reach the finish reward and push-off-goal.

## Deliberate, behaviour-preserving deviations from the reference
1. Rewards are integer units of 0.1 (exact sums for return/RTG); float reward is `units/10`.
2. Initial boxes-on-goal is computed from the board (reference hardcodes 0; same on all official levels).
3. Out-of-bounds counts as a wall (reference relies on the wall ring).
4. `truncated` = horizon reached and not solved; solving on the last step is `terminated`.

## Estimates (NOT measured; the Dell has not been inspected)
- Prompt length ~600-800 tokens (board with interior walls ~300, K=4 history ~150-200, instructions/RTG/step ~100).
- 250k examples x ~700 tokens = ~175M tokens/epoch, ~350M for 2 epochs.
- 8B BF16 weights ~16 GB: one 48 GB GPU holds the model plus LoRA, activations and optimiser state at length 1024 with gradient checkpointing, so 4-way DDP (not sharding) is the plan.
- Throughput of 2.5-3.5k tokens/s/GPU is a guess; full training would then take roughly 7-10 h. Replace with the Phase 5 benchmark.
- Closed-loop evaluation probably costs more than training; plan to score the 4 action tokens in one forward pass.

## Open decisions (need approval at the stated phase)
- **O1 (Phase 2)** Access to `AlignmentResearch/learned-planner` (DRC checkpoints, licence, action mapping, reward convention) and `AlignmentResearch/boxoban-astar-solutions` (format, whether solutions are step- or push-optimal). Neither has been inspected yet; plan is to replay everything in our adapter and trust only our rewards.
- **O2 (Phase 2)** Solver: push-based A* with reverse-BFS box-distance heuristic and dead-square / frozen-box pruning, node and time limits; solutions labelled "heuristic" unless optimality is proven under the stated cost.
- **O3 (Phase 2)** Policy-mixture sizes (pilot 600/400/500/300/200; main 50/40/10), to be revisited after pilot return-diversity statistics.
- **O4 (Phase 3)** Compact serialisation if the full natural-language board exceeds the token budget (do not replace with ASCII without approval).
- **O5 (Phase 4+)** Thinking-mode evaluation protocol, constrained-decoding policy, the first GPU run on the Dell.
- **O6 (Phase 6)** Mixture comparison 50/40/10 vs 80/15/5 vs 30/50/20 on matched small runs before the main run.
- **O7** Any reward-shaping ablation, any online RL, any scratch model: not authorised.

## Environment notes
- This development container has no GPU and cannot reach Hugging Face; everything needing a GPU or Qwen3-8B (Phase 4 onward) must run on the Dell. `boxoban-rtg hardware-check` is the first thing to run there.
- Windows graphics-memory figures are not physical VRAM; use `nvidia-smi` and `torch.cuda.mem_get_info` inside WSL2. Do not install a Linux display driver in WSL2.
