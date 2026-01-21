from typing import List, Union, Literal, Optional

import yaml
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


class CalibrationConfig(BaseModel):
    task: Literal["calibration"]
    charge_conversion: bool = True
    plot: bool = True


class FitSpecPars(BaseModel):
    xmin: float = float("-inf")
    xmax: float = float("inf")
    num_sigma_left: float = 1.5
    num_sigma_right: float = 1.5
    absolute_sigma: bool = True


class SpectrumSubtask(BaseModel):
    subtask: str
    skip: bool = False
    model: str
    fit_pars: FitSpecPars = Field(default_factory=FitSpecPars)


class SpectrumFittingConfig(BaseModel):
    task: Literal["spectrum_fitting"]
    subtasks: List[SpectrumSubtask]
    # plot: bool = True
    # plot_range: List[float] = [0.0, 10.0]


class GainConfig(BaseModel):
    task: Literal["gain"]
    w: float = 26.0
    energy: float = KALPHA
    target: Optional[str] = None
    fit: bool = True
    plot: bool = True
    label: Optional[str] = None
    yscale: Literal["linear", "log"] = "log"


class ResolutionConfig(BaseModel):
    task: Literal["resolution"]
    label: str = ""
    plot: bool = True

class GainTrendConfig(BaseModel):
    task: Literal["gain_trend"]
    time_unit: Literal["s", "m", "h"] = "h"
    subtasks: List[SpectrumSubtask]


TaskType = Union[CalibrationConfig, SpectrumFittingConfig, GainConfig, ResolutionConfig, GainTrendConfig]

class AppConfig(BaseModel):
    acquisition: Acquisition
    detector: Detector
    source: Source
    pipeline: List[TaskType]

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))
        
    @property
    def calibration(self) -> Optional[CalibrationConfig]:
        return next((t for t in self.pipeline if isinstance(t, CalibrationConfig)), None)
    
    @property
    def spectrum_fitting(self) -> Optional[SpectrumFittingConfig]:
        return next((t for t in self.pipeline if isinstance(t, SpectrumFittingConfig)), None)
    