import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.models import Fe55Forest, Gaussian
from aptapy.plotting import plt

from analysis import ANALYSIS_DATA
from analysis.fileio import DataFolder, PulsatorFile, SourceFile
from analysis.utils import KALPHA, Detector

from uncertainties import unumpy
import uncertainties
folder_path = ANALYSIS_DATA / "251118"
folder = DataFolder(folder_path)

SIGMA_LEFT = 1.5
SIGMA_RIGHT = 1.

def estimate_gain(folder, model: AbstractFitModel):
    folder = DataFolder(folder)
    detector = Detector('Ar', 1e-12)
    pulses = PulsatorFile(folder.pulse_files[0])
    line_model = pulses.analyze_pulse()

    source_files = [SourceFile(_s) for _s in folder.source_files]
    voltages = np.array([source.voltage for source in source_files])
    if issubclass(model, Fe55Forest):
        pars_Fe = np.array([source.fit_line_forest(SIGMA_LEFT, SIGMA_RIGHT) for source in source_files])
        line_ADC = KALPHA / pars_Fe[:, 2]
    elif issubclass(model, Gaussian):
        pars = np.array([source.fit_line(num_sigma_left=SIGMA_LEFT, num_sigma_right=SIGMA_RIGHT) for source in source_files])
        line_ADC = pars[:, 1]
    else:
        raise ValueError("Pass a valid class")

    if folder.folder_path.name == "251118":
        voltages = np.append(voltages, [300., 310., 320.])
        line_ADC = np.append(line_ADC, [12., 16., 21.])
    if folder.folder_path.name == "251127":
        mask = voltages == 350.
        line_350 = line_ADC[mask]
        line_ADC = line_ADC[np.logical_not(mask)]
        voltages = voltages[np.logical_not(mask)]
        line_ADC = np.append(line_ADC, np.min(line_350))
        voltages = np.append(voltages, 350.)
    gain = detector.gain(line_model, voltages, line_ADC)
    
    return voltages, gain

def estimate_gains():
    detector = Detector('Ar', 1e-12)
    pulses = PulsatorFile(folder.pulse_files[0])
    line_model = pulses.analyze_pulse()

    source_files = [SourceFile(_s) for _s in folder.source_files]
    voltages = np.array([source.voltage for source in source_files])
    # Fe55
    pars_Fe = np.array([source.fit_line_forest(SIGMA_LEFT, SIGMA_RIGHT) for source in source_files])
    line_ADC_Fe = KALPHA / pars_Fe[:, 2]
    # Gauss
    models = [source.fit_line(num_sigma_left=SIGMA_LEFT, num_sigma_right=SIGMA_RIGHT,) for source in source_files]
    line_ADC_Gauss = np.array([_model.mu.ufloat() for _model in models])
    # Plot
    voltages = np.append(voltages, [300, 310, 320])
    line_ADC_Fe = np.append(line_ADC_Fe, [12, 16, 21])
    line_ADC_Gauss = np.append(line_ADC_Gauss, [12, 16, 21])

    gain_Fe = detector.gain(line_model, voltages, line_ADC_Fe)
    gain_Gauss = detector.gain(line_model, voltages, line_ADC_Gauss)
    plt.figure('Gain')
    plt.errorbar(voltages, unumpy.nominal_values(gain_Fe), unumpy.std_devs(gain_Fe), fmt='ko', label="Fe55 Forest")
    plt.errorbar(voltages, unumpy.nominal_values(gain_Gauss), unumpy.std_devs(gain_Gauss), fmt='o', label="Single Gaussian")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

def estimate_resolution(model: AbstractFitModel):
    detector = Detector('Ar', 1e-12)
    source_files = [SourceFile(_s) for _s in folder.source_files]
    voltages = np.array([source.voltage for source in source_files])
    if issubclass(model, Fe55Forest):
        pars_Fe = np.array([source.fit_line_forest(SIGMA_LEFT, SIGMA_RIGHT) for source in source_files])
        line_ADC = KALPHA / pars_Fe[:, 2]
        sigma = pars_Fe[:, 3]
    elif issubclass(model, Gaussian):
        pars = np.array([source.fit_line(num_sigma_left=SIGMA_LEFT, num_sigma_right=SIGMA_RIGHT) for source in source_files])
        line_ADC = pars[:, 1]
        sigma = pars[:, 2]
    else:
        raise ValueError("Pass a valid class")
    resolution = detector.energy_resolution(voltages, line_ADC, sigma)
    print(resolution)

def estimate_resolutions():
    detector = Detector('Ar', 1e-12)
    source_files = [SourceFile(_s) for _s in folder.source_files]
    voltages = np.array([source.voltage for source in source_files])
    # Fe55
    pars_Fe = np.array([source.fit_line_forest(SIGMA_LEFT, SIGMA_RIGHT) for source in source_files])
    # line_ADC_Fe = np.array([KALPHA / _model.energy_scale.ufloat() for _model in models_Fe])
    line_ADC_Fe = KALPHA / pars_Fe[:, 2]
    # amplitude_ratio_Fe = np.array([_model.amplitude0.ufloat() / _model.amplitude1.ufloat() for _model in models_Fe])
    amplitude_ratio_Fe = pars_Fe[:, 0] / pars_Fe[:, 1]
    print(amplitude_ratio_Fe)

    # # Gauss
    # models_Gauss = [source.fit_line(num_sigma_left=SIGMA_LEFT, num_sigma_right=SIGMA_RIGHT,) for source in source_files]
    # line_ADC_Gauss = np.array([_model.mu.ufloat() for _model in models_Gauss])
    
    sigma_Fe = pars_Fe[:, 3]
    resolution_Fe = detector.energy_resolution(voltages, line_ADC_Fe, sigma_Fe)

    # sigma_Gauss = np.array([_model.sigma.ufloat() for _model in models_Gauss])
    # resolution_Gauss = detector.energy_resolution(voltages, line_ADC_Gauss, sigma_Gauss)

    plt.figure("Energy Resolution")
    plt.errorbar(voltages, unumpy.nominal_values(resolution_Fe), unumpy.std_devs(resolution_Fe),
        fmt='ko', label="Fe55 Forest")
    # plt.errorbar(voltages, unumpy.nominal_values(resolution_Gauss), unumpy.std_devs(resolution_Gauss),
    #     fmt='o', label="Single Gaussian")
    plt.xlabel("Voltage [V]")
    plt.ylabel("FWHM/E")
    plt.legend()

    plt.figure("Amplitude ratio Ka / Kb Fe55Forest")
    plt.errorbar(voltages, unumpy.nominal_values(amplitude_ratio_Fe), unumpy.std_devs(amplitude_ratio_Fe),
                fmt='ko', label="Fe55 Forest")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Ka / Kb")
    plt.legend()

def compare_gain(folder0, folder1):
    v0, g0 = estimate_gain(folder0, Gaussian)
    v1, g1 = estimate_gain(folder1, Gaussian)

    plt.errorbar(v0, unumpy.nominal_values(g0), unumpy.std_devs(g0), fmt='ok', label='W2b 86.6 top-left')
    plt.errorbar(v1, unumpy.nominal_values(g1), unumpy.std_devs(g1), fmt='ob', label='W8b 86.6 top-right high rate')

    plt.legend()
# estimate_gain(Gaussian)
# estimate_gain(Fe55Forest)
# estimate_resolution(Fe55Forest)
# estimate_resolution(Fe55Forest)
# estimate_gains()
# estimate_resolutions()

compare_gain(ANALYSIS_DATA / "251118", ANALYSIS_DATA / "251127")

plt.show()
