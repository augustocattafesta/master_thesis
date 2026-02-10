"""Configuration models for the analysis application."""
import pathlib
from dataclasses import dataclass
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .utils import KALPHA


class Acquisition(BaseModel):
    date: str | None = None
    chip: str | None = None
    structure: str | None = None
    gas: str | None = None
    w: float = 26.0
    element: str | None = None
    e_peak: float = KALPHA


@dataclass(frozen=True)
class CalibrationDefaults:
    charge_conversion: bool = True
    show: bool = True


class CalibrationConfig(BaseModel):
    task: Literal["calibration"]
    charge_conversion: bool = CalibrationDefaults.charge_conversion
    show: bool = CalibrationDefaults.show


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
    target: str
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
    show: bool = True


class GainConfig(BaseModel):
    task: Literal["gain"]
    target: str
    w: float = GainDefaults.w
    energy: float = GainDefaults.energy
    fit: bool = GainDefaults.fit
    show: bool = GainDefaults.show


class GainTrendConfig(BaseModel):
    task: Literal["gain_trend"]
    target: str
    w: float = GainDefaults.w
    energy: float = GainDefaults.energy
    # time_unit: Literal["s", "m", "h"] = "h"
    subtasks: list[FitSubtask] | None = Field(default=None)


@dataclass(frozen=True)
class GainCompareDefaults:
    combine: list[str] = Field(default_factory=list)
    label: str | None = None
    show: bool = True


class GainCompareConfig(BaseModel):
    task: Literal["compare_gain"]
    target: str
    combine: list[str] = GainCompareDefaults.combine


class TrendCompareConfig(BaseModel):
    task: Literal["compare_trend"]
    target: str


@dataclass(frozen=True)
class ResolutionDefaults:
    show: bool = True
    title: str | None = None
    label: str | None = None


class ResolutionConfig(BaseModel):
    task: Literal["resolution"]
    target: str
    show: bool = ResolutionDefaults.show


class ResolutionEscapeConfig(BaseModel):
    task: Literal["resolution_escape"]
    target_main: str
    target_escape: str
    show: bool = ResolutionDefaults.show
    label: str | None = ResolutionDefaults.label


@dataclass(frozen=True)
class ResolutionCompareDefaults:
    combine: list[str] = Field(default_factory=list)
    show: bool = True


class ResolutionCompareConfig(BaseModel):
    task: Literal["compare_resolution"]
    target: str
    combine: list[str] = ResolutionCompareDefaults.combine
    show: bool = ResolutionCompareDefaults.show


@dataclass(frozen=True)
class DriftDefaults:
    rate: bool = False
    threshold: float = 1.5
    show: bool = True
    label: str | None = None
    yscale: Literal["linear", "log"] = "linear"


class DriftConfig(BaseModel):
    task: Literal["drift"]
    target: str
    rate: bool = DriftDefaults.rate
    w: float = GainDefaults.w
    energy: float = GainDefaults.energy
    threshold: float = DriftDefaults.threshold
    show: bool = DriftDefaults.show
    label: str | None = DriftDefaults.label
    yscale: Literal["linear", "log"] = DriftDefaults.yscale


@dataclass(frozen=True)
class PlotDefaults:
    title: str | None = None
    label: str | None = None
    task_labels: list[str] | None = None
    loc: str = "best"
    xrange: list[float] | None = Field(None, min_length=2, max_length=2)
    xmin_factor: float = 1.
    xmax_factor: float = 1.
    voltage: bool = False
    show: bool = True


class PlotConfig(BaseModel):
    task: Literal["plot"]
    targets: list[str] | None = None
    title: str | None = PlotDefaults.title
    label: str | None = PlotDefaults.label
    task_labels: list[str] | None = PlotDefaults.task_labels
    loc: str = PlotDefaults.loc
    xrange: list[float] | None = PlotDefaults.xrange
    xmin_factor: float = PlotDefaults.xmin_factor
    xmax_factor: float = PlotDefaults.xmax_factor
    voltage: bool = PlotDefaults.voltage
    show: bool = PlotDefaults.show


@dataclass(frozen=True)
class PlotStyleDefaults:
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "linear"
    title: str | None = None
    label: str = "Data"
    legend_label: str | None = None
    legend_loc: str = "best"
    marker: str = "."
    linestyle: str = "-"
    color: str | None = None
    fit_output: bool = False
    annotate_min: bool = False


class PlotStyleConfig(BaseModel):
    xscale: Literal["linear", "log"] = PlotStyleDefaults.xscale
    yscale: Literal["linear", "log"] = PlotStyleDefaults.yscale
    title: str | None = PlotStyleDefaults.title
    label: str | None = PlotStyleDefaults.label
    legend_label: str | None = PlotStyleDefaults.legend_label
    legend_loc: str = PlotStyleDefaults.legend_loc
    marker: str = PlotStyleDefaults.marker
    linestyle: str = PlotStyleDefaults.linestyle
    color: str | None = PlotStyleDefaults.color
    fit_output: bool = PlotStyleDefaults.fit_output
    annotate_min: bool = PlotStyleDefaults.annotate_min


class StyleConfig(BaseModel):
    tasks: dict[str, PlotStyleConfig] = Field(default_factory=dict)
    folders: dict[str, PlotStyleConfig] = Field(default_factory=dict)



TaskType = CalibrationConfig | FitSpecConfig | GainConfig | ResolutionConfig | \
    ResolutionEscapeConfig | GainTrendConfig | PlotConfig | DriftConfig | GainCompareConfig | \
    ResolutionCompareConfig | TrendCompareConfig

class AppConfig(BaseModel):
    acquisition: Acquisition = Field(default_factory=Acquisition)
    pipeline: list[TaskType]
    style: StyleConfig = Field(default_factory=StyleConfig)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "AppConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str | pathlib.Path) -> None:
        data = self.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @property
    def calibration(self) -> CalibrationConfig | None:
        return next((t for t in self.pipeline if isinstance(t, CalibrationConfig)), None)

    @property
    def fit_spec(self) -> FitSpecConfig | None:
        return next((t for t in self.pipeline if isinstance(t, FitSpecConfig)), None)

    @property
    def plot(self) -> PlotConfig | None:
        return next((t for t in self.pipeline if isinstance(t, PlotConfig)), None)
