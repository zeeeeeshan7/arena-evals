"""Pure statistics: no I/O."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return {"point": self.point, "ci95": [self.lo, self.hi]}
