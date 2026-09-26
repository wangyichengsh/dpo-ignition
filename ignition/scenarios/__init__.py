from .base import Problem, Scenario
from .chips import Chips
from .circles import Circles
from .torus import Torus

SCENARIOS = {s.name: s for s in (Chips, Circles, Torus)}


def get_scenario(name):
    if name not in SCENARIOS:
        raise SystemExit(f"❌ 未知场景 {name!r}, 可选: {list(SCENARIOS)}")
    sc = SCENARIOS[name]()
    sc.self_check()
    return sc


__all__ = ["Problem", "Scenario", "SCENARIOS", "get_scenario"]
