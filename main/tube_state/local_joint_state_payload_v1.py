"""Carrier-agnostic interfaces for a local joint state/payload candidate.

This module deliberately contains no carrier, code, budget, threshold, path
decoder, or decision rule.  It only keeps the two time axes distinct and
defines records that an adopted writer or public receiver may fill later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


COMPONENT_STATUSES = ("SCORED", "MISSING", "FAILED")
PATH_STATUSES = ("UNSCORED", "CANDIDATE", "UNRESOLVED")


@dataclass(frozen=True)
class SamplingStepCoord:
    """One coordinate on the generation/sampling axis."""

    index: int
    total_steps: int
    sigma: float

    def __post_init__(self) -> None:
        if type(self.index) is not int or type(self.total_steps) is not int:
            raise TypeError("sampling indices must be integers")
        if self.total_steps <= 0 or not 0 <= self.index < self.total_steps:
            raise ValueError("sampling index outside the declared history")
        if not math.isfinite(float(self.sigma)) or float(self.sigma) <= 0:
            raise ValueError("sampling sigma must be finite and positive")


@dataclass(frozen=True)
class VideoWindowCoord:
    """One local support on the output/received-video axis.

    ``coordinate_space`` and ``layer`` are explicit because a candidate may
    observe RGB windows while another diagnostic may use a latent layer.  The
    coordinate is never inferred from a sampling-step index.
    """

    window_id: str
    time_indices: tuple[int, ...]
    coordinate_space: str
    layer: str
    # Public order is (y0, y1, x0, x1), with half-open bounds.
    spatial_region: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not self.window_id or not self.coordinate_space or not self.layer:
            raise ValueError("window id, coordinate space, and layer are required")
        if not self.time_indices or any(type(x) is not int or x < 0 for x in self.time_indices):
            raise ValueError("video-window time coordinates must be nonnegative integers")
        if len(set(self.time_indices)) != len(self.time_indices):
            raise ValueError("video-window time coordinates must be unique")
        if self.spatial_region is not None:
            if (len(self.spatial_region) != 4 or
                    any(type(x) is not int or x < 0 for x in self.spatial_region)):
                raise ValueError("spatial region must be four nonnegative integers")
            y0,y1,x0,x1 = self.spatial_region
            if not y0 < y1 or not x0 < x1:
                raise ValueError("spatial region must have positive area")


@dataclass(frozen=True)
class JointControlRequest:
    """Metadata passed to an externally supplied joint writer component."""

    sampling: SamplingStepCoord
    windows: tuple[VideoWindowCoord, ...]
    state_spec: Any
    payload_spec: Any
    requested_budget: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (not self.windows or
                any(not isinstance(x,VideoWindowCoord) for x in self.windows) or
                len({x.window_id for x in self.windows}) != len(self.windows)):
            raise ValueError("joint control requires distinct local video windows")
        if not isinstance(self.requested_budget, Mapping):
            raise TypeError("requested budget must be an explicit mapping")
        for key,value in self.requested_budget.items():
            if not isinstance(key,str) or not isinstance(value,(str,int,float,bool,type(None))):
                raise TypeError("requested budget records must be flat scalar descriptions")
            if isinstance(value,float) and not math.isfinite(value):
                raise ValueError("requested budget numbers must be finite")


@dataclass(frozen=True)
class JointControlResult:
    """Writer output in clean-estimate coordinates.

    The provider always supplies ``joint_delta``, the only applied value.
    State and payload deltas are optional diagnostics; this interface does not
    assume addition, weighting, or any particular cap between them.
    """

    joint_delta: Any
    composition: str
    enabled: bool
    state_delta: Any | None = None
    payload_delta: Any | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.composition:
            raise ValueError("the external provider must name its composition")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be an explicit boolean")
        if not isinstance(self.detail, Mapping):
            raise TypeError("joint-control detail must be a mapping")


@dataclass(frozen=True)
class WindowObservation:
    """Truth-free soft evidence retained for exactly one requested window.

    State and payload have independent statuses so one missing component does
    not erase the other component's observed values.
    """

    window: VideoWindowCoord
    state_status: str
    payload_status: str
    state_soft_q: tuple[float, ...] | None = None
    payload_soft_evidence: tuple[float, ...] | None = None
    state_support: Mapping[str, Any] = field(default_factory=dict)
    payload_support: Mapping[str, Any] = field(default_factory=dict)
    state_error: str | None = None
    payload_error: str | None = None

    def __post_init__(self) -> None:
        if self.state_status not in COMPONENT_STATUSES or self.payload_status not in COMPONENT_STATUSES:
            raise ValueError("unknown component observation status")
        if not isinstance(self.state_support,Mapping) or not isinstance(self.payload_support,Mapping):
            raise TypeError("component support must be a mapping")
        self._validate_component("state",self.state_status,self.state_soft_q,self.state_error)
        self._validate_component("payload",self.payload_status,self.payload_soft_evidence,self.payload_error)

    @staticmethod
    def _validate_component(name,status,values,error) -> None:
        if status == "SCORED":
            if values is None or not values or not all(math.isfinite(float(x)) for x in values):
                raise ValueError(f"scored {name} evidence must be finite and nonempty")
            if error is not None:
                raise ValueError(f"scored {name} evidence cannot carry an error")
        elif values is not None:
            raise ValueError(f"missing/failed {name} evidence cannot carry values")

    @property
    def status(self) -> str:
        statuses=(self.state_status,self.payload_status)
        if statuses == ("SCORED","SCORED"):return "SCORED"
        if "SCORED" in statuses:return "PARTIAL"
        if statuses == ("MISSING","MISSING"):return "MISSING"
        return "FAILED"


@dataclass(frozen=True)
class BoundaryRef:
    """Container for a possible boundary between two public windows."""

    boundary_id: str
    left_window_id: str
    right_window_id: str


@dataclass(frozen=True)
class PathCandidate:
    """Unscored path container; no transition or decoding rule is implied."""

    path_id: str
    window_ids: tuple[str, ...]
    boundary_ids: tuple[str, ...] = ()
    status: str = "UNSCORED"

    def __post_init__(self) -> None:
        if not self.path_id or not self.window_ids or self.status not in PATH_STATUSES:
            raise ValueError("invalid path-candidate container")


@dataclass(frozen=True)
class PathEquivalenceClass:
    """Named set of paths that a later adopted rule may treat as equivalent."""

    class_id: str
    path_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.class_id or not self.path_ids or len(set(self.path_ids)) != len(self.path_ids):
            raise ValueError("equivalence class requires distinct path ids")
