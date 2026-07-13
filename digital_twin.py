"""Timestamped digital-twin state buffer.

The environment keeps two deliberately separate views of the world:

``physical_states``
    Ground truth returned by SUMO/MockWorld.  It may only be used to execute
    and settle a decision.

``states``
    The latest snapshot visible to the decision system after the configured
    synchronization delay.  Candidate discovery and policy observations must
    use this view so experiments cannot accidentally leak future state.

The class is intentionally independent from SUMO.  Later work can replace the
fixed-delay buffer with a VD/BSM data-assimilation source without changing the
RL environment interface.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class TwinSnapshot:
    """One immutable-in-practice observation received by the digital twin."""

    observed_at: float
    received_at: float
    states: dict[str, dict[str, Any]]

    @property
    def age_s(self) -> float:
        return max(0.0, self.received_at - self.observed_at)


class DigitalTwinStateStore:
    """Maintain a delayed, timestamped view of vehicle states.

    ``delay_ticks`` is the requested synchronization delay.  During warm-up,
    the oldest available snapshot is returned, so Age of Information grows
    from zero until the configured delay is reached.
    """

    def __init__(self, delay_ticks: int = 0, dt: float = 1.0,
                 position_sigma_m: float = 1.5,
                 velocity_sigma_mps: float = 0.5):
        if int(delay_ticks) < 0:
            raise ValueError("delay_ticks must be >= 0")
        self.delay_ticks = int(delay_ticks)
        self.dt = float(dt)
        self.position_sigma_m = float(position_sigma_m)
        self.velocity_sigma_mps = float(velocity_sigma_mps)
        self._history: deque[tuple[float, dict[str, dict[str, Any]]]] = deque(
            maxlen=self.delay_ticks + 1)
        self._current = TwinSnapshot(0.0, 0.0, {})

    @staticmethod
    def _copy_states(states: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        # SUMO adapters may reuse/mutate dictionaries.  A deep copy prevents a
        # delayed snapshot from silently becoming current physical state.
        return {str(vid): deepcopy(dict(state)) for vid, state in states.items()}

    def reset(self) -> None:
        self._history.clear()
        self._current = TwinSnapshot(0.0, 0.0, {})

    def update(self, now: float,
               physical_states: Mapping[str, Mapping[str, Any]]) -> TwinSnapshot:
        self._history.append((float(now), self._copy_states(physical_states)))
        observed_at, states = self._history[0]
        self._current = TwinSnapshot(observed_at, float(now), states)
        return self._current

    @property
    def snapshot(self) -> TwinSnapshot:
        return self._current

    @property
    def states(self) -> dict[str, dict[str, Any]]:
        return self._current.states

    @property
    def age_s(self) -> float:
        return self._current.age_s

    def position_uncertainty_m(self) -> float:
        """Simple dead-reckoning uncertainty proxy for policy/QoS features."""
        return math.hypot(
            self.position_sigma_m,
            self.velocity_sigma_mps * self.age_s,
        )

