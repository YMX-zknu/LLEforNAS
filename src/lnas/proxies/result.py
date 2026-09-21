from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ProxyResult:
    name: str
    score: float
    raw_value: float
    stable: bool
    details: Dict[str, float]

    def to_dict(self):
        return asdict(self)
