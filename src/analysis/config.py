"""Configuration models for the analysis application."""
import pathlib
from dataclasses import dataclass
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .utils import KALPHA


class Acquisition(BaseModel):
    date: str
    chip: str
    structure: str


class Detector(BaseModel):
    gas: str
    w: float = 26.0


class Source(BaseModel):
    element: str
    e_peak: float = KALPHA


@dataclass(frozen=True)
class CalibrationDefaults:
    charge_conversion: bool = True
    plot: bool = True


class CalibrationConfig(BaseModel):
    task: Literal["calibration"]
    charge_conversion: bool = CalibrationDefaults.charge_conversion
    plot: bool = CalibrationDefaults.plot


@dataclass(frozen=True)
class FitPeakDefaults:
    xmin: float = float("-inf")
    xmax: float = float("inf")
    num_sigma_left: float = 1.5
    num_sigma_right: float = 1.5
    absolute_sigma: bool = True
    p0: list[float] | None = None


class FitPars(BaseModel):
    xmin: float | None = FitPeakDefaults.xmin
    xmax: float | None = FitPeakDefaults.xmax
    num_sigma_left: float | None = FitPeakDefaults.num_sigma_left
    num_sigma_right: float | None = FitPeakDefaults.num_sigma_right
    absolute_sigma: bool | None = FitPeakDefaults.absolute_sigma
    p0: list[float] | None = FitPeakDefaults.p0


class FitSubtask(BaseModel):
    subtask: str
    skip: bool = False
    model: str
    fit_pars: FitPars = Field(default_factory=FitPars)


class FitSpecConfig(BaseModel):
    task: Literal["fit_spec"]
    subtasks: list[FitSubtask]


@dataclass(frozen=True)
class GainDefaults:
    w: float = 26.0
    energy: float = KALPHA
    fit: bool = True
    plot: bool = True
    label: str | None = None
    yscale: Literal["linear", "log"] = "log"


class GainConfig(BaseModel):
    task: Literal["gain"]
    target: str
    w: float = GainDefaults.w
    energy: float = GainDefaults.energy
    fit: bool = GainDefaults.fit
    plot: bool = GainDefaults.plot
    label: str | None = GainDefaults.label
    yscale: Literal["linear", "log"] = GainDefaults.yscale


class GainTrendConfig(BaseModel):
    task: Literal["gain_trend"]
    target: str | None
    w: float = GainDefaults.w
    energy: float = GainDefaults.energy
    # time_unit: Literal["s", "m", "h"] = "h"
    subtasks: list[FitSubtask] | None = Field(default=None)


class GainCompareConfig(BaseModel):
    task: Literal["compare_gain"]
    aggregate: bool = False
    label: str | None = GainDefaults.label
    yscale: Literal["linear", "log"] = GainDefaults.yscale


@dataclass(frozen=True)
class ResolutionDefaults:
    plot: bool = True
    label: str | None = None


class ResolutionConfig(BaseModel):
    task: Literal["resolution"]
    target: str
    plot: bool = ResolutionDefaults.plot
    label: str | None = ResolutionDefaults.label


class ResolutionEscapeConfig(BaseModel):
    task: Literal["resolution_escape"]
    target_main: str
    target_escape: str


@dataclass(frozen=True)
class DriftDefaults:
    rate: bool = False
    threshold: float = 1.5
    plot: bool = True
    label: str | None = None
    yscale: Literal["linear", "log"] = "linear"


class DriftConfig(BaseModel):
    task: Literal["drift"]
    target: str | None = None
    rate: bool = DriftDefaults.rate
    w: float = GainDefaults.w
    energy: float = GainDefaults.energy
    threshold: float = DriftDefaults.threshold
    plot: bool = DriftDefaults.plot
    label: str | None = DriftDefaults.label
    yscale: Literal["linear", "log"] = DriftDefaults.yscale


@dataclass(frozen=True)
class PlotDefaults:
    plot: bool = True
    xrange: list[float] | None = Field(None, min_length=2, max_length=2)
    label: str = ""
    task_labels: list[str] | None = None
    loc: str = "best"


class PlotConfig(BaseModel):
    task: Literal["plot"]
    targets: list[str] | None = None
    xrange: list[float] | None = PlotDefaults.xrange
    label: str | None = PlotDefaults.label
    task_labels: list[str] | None = PlotDefaults.task_labels
    loc: str = PlotDefaults.loc



TaskType = CalibrationConfig | FitSpecConfig | GainConfig | ResolutionConfig | \
    ResolutionEscapeConfig | GainTrendConfig | PlotConfig | DriftConfig | GainCompareConfig

class AppConfig(BaseModel):
    acquisition: Acquisition
    detector: Detector
    source: Source
    pipeline: list[TaskType]

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "AppConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    @property
    def calibration(self) -> CalibrationConfig | None:
        return next((t for t in self.pipeline if isinstance(t, CalibrationConfig)), None)

    @property
    def fit_spec(self) -> FitSpecConfig | None:
        return next((t for t in self.pipeline if isinstance(t, FitSpecConfig)), None)

    @property
    def plot(self) -> PlotConfig | None:
        return next((t for t in self.pipeline if isinstance(t, PlotConfig)), None)
