"""Site geometry: where the geophones are and which way is downslope."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROAD_Y = -30.0  # the road runs east-west across the foot of the gully


@dataclass(frozen=True)
class Station:
    name: str
    x: float  # metres east
    y: float  # metres north


@dataclass(frozen=True)
class Site:
    name: str
    stations: tuple[Station, ...]
    downslope: tuple[float, float] = (0.0, -1.0)  # unit vector of steepest descent

    def xy(self, names: list[str] | None = None) -> np.ndarray:
        by_name = {s.name: s for s in self.stations}
        sel = [by_name[n] for n in names] if names is not None else list(self.stations)
        return np.array([[s.x, s.y] for s in sel], dtype=float)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.stations]


def demo_site() -> Site:
    """Six geophones zig-zagging down a 250 m gully; a road runs across the foot (y = -30 m)."""
    stations = tuple(Station(f"G{i + 1}", 20.0 if i % 2 else -20.0, 250.0 - 50.0 * i) for i in range(6))
    return Site("demo-gully", stations)
