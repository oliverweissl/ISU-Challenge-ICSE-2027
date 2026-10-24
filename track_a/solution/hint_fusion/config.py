from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("sum", "mean", "region_gated", "phase_gated", "conflict_aware")


@dataclass
class FusionConfig:
    mode: str = "mean"
    modalities: tuple = ("canny", "depth")
    # sum/mean: static per-modality weights (sum: raw, mean: normalised to sum 1)
    weights: dict = field(default_factory=lambda: {"canny": 1.0, "depth": 0.2})
    # region_gated
    gate_normalize: bool = False  # renormalise per-token gates to sum 1
    gate_table: str = str(HERE / "gate_table.json")
    seg_class_map: str = ""  # default: data/instance_seg/instance_seg_class_map.json
    # phase_gated: weight of each modality goes linearly from (start) at t=0 (noise) to (end) at t=1 (clean),
    # t = pipeline-normalised timestep; 'step' switches at phase_switch instead of interpolating
    phase_start: dict = field(default_factory=lambda: {"canny": 0.6, "depth": 0.3})
    phase_end: dict = field(default_factory=lambda: {"canny": 1.0, "depth": 0.1})
    phase_shape: str = "linear"  # linear | step
    phase_switch: float = 0.5
    phase_normalize: bool = False  # renormalise weights to sum 1
    # conflict_aware: token-level cos < conflict_thresh -> project each hint off the other
    conflict_base: str = "sum"  # how to combine after projection: mean | sum
    conflict_thresh: float = 0.0
    conflict_center: bool = True  # subtract per-layer token-mean (common mode) before cos/projection
    conflict_strength: float = 1.0  # 1 = full projection

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "modalities" in d:
            d["modalities"] = tuple(d["modalities"])
        return cls(**d)
