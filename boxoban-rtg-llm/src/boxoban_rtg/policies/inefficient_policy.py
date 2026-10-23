from __future__ import annotations

from ..environments.boxoban import replay
from ..solvers.search import SolverConfig, solve
from .base import Policy


class InefficientSolverPolicy(Policy):
    """Policy C: genuine but suboptimal solutions, from alternative search plans (no padding with detours).

    Mechanisms (chosen by `mechanism` or by the episode RNG when "mixed"):
      weighted    weighted A* (f = g + w*h, w log-uniform in [3, 16]) -> valid but longer plans (w=2 barely
                  differs from optimal: measured +0.17 steps; w=8: +5.6 steps)
      pushes      push-optimal plan (objective="pushes"): fewest pushes, more walking; measured +13 steps on average
      noisy       A* with deterministic heuristic noise -> different box-pushing orders / routes
      banned_push the optimal plan's first push is forbidden -> a different opening strategy
    The plan is followed exactly; replayed return = 14 - 0.1 * steps for solved official levels, so lower
    returns here come from genuinely longer plans. info() reports the mechanism and settings.
    """

    policy_id = "inefficient_solver"

    MECHANISMS = ("weighted", "noisy", "banned_push", "pushes")

    def _config(self, level, rng, mech, base):
        if mech == "weighted":
            return SolverConfig(weight=3.0 * (16.0 / 3.0) ** rng.random(), **base)
        if mech == "pushes":
            return SolverConfig(objective="pushes", **base)
        if mech == "noisy":
            return SolverConfig(noise=rng.uniform(0.5, 2.0), noise_seed=rng.randrange(1 << 30), weight=rng.uniform(1.0, 2.0), **base)
        if mech == "banned_push":
            opt = solve(level, None, SolverConfig(**base))
            banned = frozenset()
            if opt.found:
                rep = replay(level, opt.actions)
                for a, s in zip(opt.actions, rep.steps):
                    if s.moved_box:  # the player now stands where the pushed box was
                        banned = frozenset({(s.state.player, a.delta)})
                        break
            return SolverConfig(banned_root_pushes=banned, **base)
        raise ValueError(f"unknown mechanism {mech}")

    def start(self, level, env, rng):
        super().start(level, env, rng)
        want = self.params.get("mechanism", "mixed")
        base = dict(max_nodes=self.params.get("max_nodes", 100_000), time_limit_s=self.params.get("time_limit_s", 10.0))
        order = list(self.MECHANISMS) if want == "mixed" else [want]
        if want == "mixed":
            rng.shuffle(order)
        self.attempts = []
        for mech in order:  # in mixed mode, a mechanism that finds no plan (e.g. a forced opening was banned) is replaced
            cfg = self._config(level, rng, mech, base)
            result = solve(level, None, cfg)
            self.attempts.append((mech, result.status))
            if result.found:
                break
        self.mech, self.cfg, self.result = mech, cfg, result
        self.plan, self.i = list(result.actions), 0

    def act(self, state):
        if self.i < len(self.plan):
            a = self.plan[self.i]
            self.i += 1
            return a
        from ..environments.actions import Action

        return Action(self.rng.randrange(4))

    def info(self):
        r = self.result
        return dict(mechanism=self.mech, solve_status=r.status, plan_steps=r.steps, plan_pushes=r.pushes,
                    nodes_expanded=r.nodes_expanded, weight=self.cfg.weight, noise=self.cfg.noise, attempts=self.attempts)
