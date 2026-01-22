"""Configuration models for the analysis application."""
from dataclasses import dataclass
from typing import Literal

import aptapy.models
import yaml
from aptapy.modeling import AbstractFitModel
from pydantic import BaseModel, Field

from .utils import KALPHA


class Acquisition(BaseModel):
    date: str
    chip: str
    structure: str
    drift: str | float
    back: str | float


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
    model_class: AbstractFitModel = aptapy.models.Gaussian
    xmin: float = float("-inf")
    xmax: float = float("inf")
    num_sigma_left: float = 1.5
    num_sigma_right: float = 1.5
    absolute_sigma: bool = True


class FitSpecPars(BaseModel):
    xmin: float | None = FitPeakDefaults.xmin
    xmax: float | None = FitPeakDefaults.xmax
    num_sigma_left: float | None = FitPeakDefaults.num_sigma_left
    num_sigma_right: float | None = FitPeakDefaults.num_sigma_right
    absolute_sigma: bool | None = FitPeakDefaults.absolute_sigma


class SpectrumSubtask(BaseModel):
    subtask: str
    skip: bool = False
    model: str
    fit_pars: FitSpecPars = Field(default_factory=FitSpecPars)


class SpectrumFittingConfig(BaseModel):
    task: Literal["spectrum_fitting"]
    subtasks: list[SpectrumSubtask]


@dataclass(frozen=True)
class GainDefaults:
    w: float = 26.0
    energy: float = KALPHA
    fit: bool = True
    plot: bool = True
    label: str | None = None
    yscale: str = "log"


class GainConfig(BaseModel):
    task: Literal["gain"]
    target: str | None = None
    w: float = GainDefaults.w
    energy: float = GainDefaults.energy
    fit: bool = GainDefaults.fit
    plot: bool = GainDefaults.plot
    label: str | None = GainDefaults.label
    yscale: Literal["linear", "log"] = GainDefaults.yscale


@dataclass(frozen=True)
class ResolutionDefaults:
    plot: bool = True
    label: str | None = None


class ResolutionConfig(BaseModel):
    task: Literal["resolution"]
    target: str | None = None
    plot: bool = ResolutionDefaults.plot
    label: str = ResolutionDefaults.label


class ResolutionEscapeConfig(BaseModel):
    task: Literal["resolution_escape"]
    target_main: str | None = None
    target_escape: str | None = None


class DriftRateConfig(BaseModel):
    task: Literal["rate"]
    target: str | None = None
    energy: float = KALPHA
    threshold: float = 0.1
    plot: bool = True
    label: str | None = None


class GainTrendConfig(BaseModel):
    task: Literal["gain_trend"]
    time_unit: Literal["s", "m", "h"] = "h"
    subtasks: list[SpectrumSubtask]


@dataclass(frozen=True)
class PlotDefaults:
    plot: bool = True
    label: str = ""
    xrange: list[float] | None = Field(None, min_length=2, max_length=2)
    task_labels: list[str] | None = None


class PlotConfig(BaseModel):
    task: Literal["plot"]
    targets: list[str] | None = None
    label: str | None = PlotDefaults.label
    xrange: list[float] | None = PlotDefaults.xrange
    task_labels: list[str] | None = PlotDefaults.task_labels


TaskType = CalibrationConfig | SpectrumFittingConfig | GainConfig | ResolutionConfig | \
    ResolutionEscapeConfig | GainTrendConfig | PlotConfig | DriftRateConfig

class AppConfig(BaseModel):
    acquisition: Acquisition
    detector: Detector
    source: Source
    pipeline: list[TaskType]

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))
        
    @property
    def calibration(self) -> CalibrationConfig | None:
        return next((t for t in self.pipeline if isinstance(t, CalibrationConfig)), None)
    
    @property
    def spectrum_fitting(self) -> SpectrumFittingConfig | None:
        return next((t for t in self.pipeline if isinstance(t, SpectrumFittingConfig)), None)
    
    @property
    def plot(self) -> PlotConfig | None:
        return next((t for t in self.pipeline if isinstance(t, PlotConfig)), None)