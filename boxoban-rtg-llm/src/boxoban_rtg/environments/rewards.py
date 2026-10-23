"""Original Gym-Sokoban reward, in exact integer units of 0.1.

Reference (gym_sokoban 0.0.6 SokobanEnv._calc_reward), evaluated after the move:
  -0.1 every step (including blocked steps)
  +1   if the number of boxes on goals increased
  -1   if it decreased
  +10  if all boxes are on goals
Integer units keep sums (episode return, RTG) free of float drift; the float
reward is units / 10.
"""
STEP_UNITS = -1
BOX_ON_UNITS = 10
BOX_OFF_UNITS = -10
FINISH_UNITS = 100
UNIT = 0.1


def reward_units(boxes_on_before: int, boxes_on_after: int, solved: bool) -> int:
    r = STEP_UNITS
    if boxes_on_after > boxes_on_before:
        r += BOX_ON_UNITS
    elif boxes_on_after < boxes_on_before:
        r += BOX_OFF_UNITS
    if solved:
        r += FINISH_UNITS
    return r


def units_to_float(u: int) -> float:
    return u / 10
