import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.models import Fe55Forest, Gaussian
from aptapy.plotting import plt

from analysis import ANALYSIS_DATA
from analysis.fileio import DataFolder, PulsatorFile, SourceFile
from analysis.utils import KALPHA, Detector

folder_path = ANALYSIS_DATA / "251118"
folder = DataFolder(folder_path)

def estimate_gain(model: AbstractFitModel):
    detector = Detector('Ar', 1e-12)
    pulses = PulsatorFile(folder.pulse_files[1])
    line_model = pulses.analyze_pulse()

    source_files = [SourceFile(_s) for _s in folder.source_files]
    voltages = np.array([source.voltage for source in source_files])
    if issubclass(model, Fe55Forest):
        models = [source.fit_line_forest(1.5, 3.) for source in source_files]
        line_ADC = np.array([KALPHA / _model.energy_scale.ufloat() for _model in models])
    elif issubclass(model, Gaussian):
        models = [source.fit_line() for source in source_files]
        line_ADC = np.array([_model.mu.ufloat() for _model in models])
    else:
        raise ValueError("Pass a valid class")

    gain = detector.gain(line_model, voltages, line_ADC)
    print(gain)

def estimate_resolution(model: AbstractFitModel):
    detector = Detector('Ar', 1e-12)
    source_files = [SourceFile(_s) for _s in folder.source_files]
    voltages = np.array([source.voltage for source in source_files])
    if issubclass(model, Fe55Forest):
        models = [source.fit_line_forest(1.5, 3.) for source in source_files]
        line_ADC = np.array([KALPHA / _model.energy_scale.ufloat() for _model in models])
    elif issubclass(model, Gaussian):
        models = [source.fit_line() for source in source_files]
        line_ADC = np.array([_model.mu.ufloat() for _model in models])
    else:
        raise ValueError("Pass a valid class")
    sigma = np.array([_model.sigma.ufloat() for _model in models])
    resolution = detector.energy_resolution(voltages, line_ADC, sigma)
    print(resolution)




# estimate_gain(Gaussian)
# estimate_gain(Fe55Forest)
estimate_resolution(Gaussian)
plt.show()
