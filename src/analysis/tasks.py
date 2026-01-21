from dataclasses import dataclass

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import last_line_color, plt
from uncertainties import unumpy

from .fileio import PulsatorFile, SourceFile, load_label
from .utils import find_peaks_iterative, gain, energy_resolution, KALPHA


@dataclass(frozen=True)
class CalibrationDefaults:
    charge_conversion: bool = True
    plot: bool = True


def calibration(
          pulse: PulsatorFile,
          charge_conversion: bool = CalibrationDefaults.charge_conversion,
          plot: bool = CalibrationDefaults.plot
      ) -> tuple[AbstractFitModel, plt.Figure, plt.Figure]:
    """
    """
    hist = pulse.hist
    pulse_fig = plt.figure(pulse.file_path.name)
    hist.plot()
    # Find pulses and fit them with Gaussian models to find their positions
    xpeaks, _ = find_peaks_iterative(hist.bin_centers(), hist.content, pulse.num_pulses)
    mu_peak = np.zeros(pulse.num_pulses, dtype=object)
    for i, xpeak in enumerate(xpeaks):
            peak_model = aptapy.models.Gaussian()
            xmin = xpeak - np.sqrt(xpeak)
            xmax = xpeak + np.sqrt(xpeak)
            peak_model.fit_iterative(hist, xmin=xmin, xmax=xmax, absolute_sigma=True)
            mu_peak[i] = peak_model.mu.ufloat()
            peak_model.plot(fit_output=True)
    plt.legend()
    # Fit the data to find the calibration parameters
    ylabel = "Charge [fC]" if charge_conversion else "Voltage [mV]"
    model = aptapy.models.Line("Calibration", "ADC Channel", ylabel)
    xdata = unumpy.nominal_values(mu_peak)
    ydata = pulse.voltage
    model.fit(xdata, ydata)
    cal_fig = plt.figure("Calibration")
    plt.errorbar(xdata, ydata, fmt=".k", label="Data")
    model.plot(fit_output=True, color=last_line_color())
    plt.legend()
    if not plot:
        plt.close(pulse_fig)
        plt.close(cal_fig)
    
    results = dict(model=model, pulse_figure=pulse_fig, calibration_figure=cal_fig)
    return results


@dataclass(frozen=True)
class FitPeakDefaults:
    model_class: AbstractFitModel = aptapy.models.Gaussian
    xmin: float = float("-inf")
    xmax: float = float("inf")
    num_sigma_left: float = 1.5
    num_sigma_right: float = 1.5
    absolute_sigma: bool = True


def fit_peak(
          source: SourceFile,
          model_class: AbstractFitModel = FitPeakDefaults.model_class,
          xmin: float = FitPeakDefaults.xmin,
          xmax: float = FitPeakDefaults.xmax,
          num_sigma_left: float = FitPeakDefaults.num_sigma_left,
          num_sigma_right: float = FitPeakDefaults.num_sigma_right,
          absolute_sigma: bool = FitPeakDefaults.absolute_sigma,
      ) -> dict:
    """
    """
    hist = source.hist
    # Without a proper initialization of xmin and xmax the fit doesn't converge
    x_peak = hist.bin_centers()[np.argmax(hist.content)]
    if xmin == float("-inf"):
        xmin = x_peak - 0.5 * np.sqrt(x_peak)
    if xmax == float("inf"):
        xmax = x_peak + 0.5 * np.sqrt(x_peak)
    # Define the dictionary of keyword arguments for the fit
    kwargs = dict(
         xmin=xmin,
         xmax=xmax,
         num_sigma_left=num_sigma_left,
         num_sigma_right=num_sigma_right,
         absolute_sigma=absolute_sigma)
    # Initialize the model and fit the data
    model = model_class()
    if isinstance(model, aptapy.models.Fe55Forest):
        model.intensity1.freeze(0.16)
    model.fit_iterative(hist, **kwargs)
    # Extract the line value and sigma from the fit results
    if isinstance(model, aptapy.models.Gaussian):
        line_val = model.status.correlated_pars[1]
        sigma = model.status.correlated_pars[2]
    elif isinstance(model, aptapy.models.Fe55Forest):
        reference_energy: float = model.energies[0]   # type: ignore [attr-defined]
        line_val = reference_energy / model.status.correlated_pars[1]
        sigma = model.status.correlated_pars[2]
    else:
        raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")
    # Plot the fit results
    model.plot(fit_output=True, color=last_line_color())
    plt.legend()
    # Return the results as a dictionary
    return dict(line_val=line_val, sigma=sigma, voltage=source.voltage, model=model)


@dataclass(frozen=True)
class GainDefaults:
    w: float = 26.0
    energy: float = KALPHA
    fit: bool = True
    plot: bool = True
    label: str = ""
    yscale: str = "log"


def gain_analysis(
        context: dict,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        fit: bool = GainDefaults.fit,
        plot: bool = GainDefaults.plot,
        label: str = GainDefaults.label,
        yscale: str = GainDefaults.yscale,
        target: str | None = None,
        ):
    """
    """
    results = context.get("results", {})
    if not results:
        return None
    if target not in results:
        line_vals = np.array([results[key]["line_val"] for key in results.keys()])
        voltage = np.array([results[key]["voltage"] for key in results.keys()])
    else:
        line_vals = results[target]["line_val"]
        voltage = results[target]["voltage"]
        fit, plot = False, False  # No fitting or plotting for single values
    gain_vals = gain(w, line_vals, energy)
    y = unumpy.nominal_values(gain_vals)
    yerr = unumpy.std_devs(gain_vals)
    fig = plt.figure("Gain vs voltage")
    if fit:
        model = aptapy.models.Exponential()
        model.fit(voltage, y, sigma=yerr, absolute_sigma=True)
        fit_label = f"Scale: {-model.scale.ufloat()} V"
        model.plot(label=fit_label, color=last_line_color())
    if label is None:
        folder_name = None
        label = load_label(folder_name)
    plt.errorbar(voltage, y, yerr=yerr, fmt=".k", label=label)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.yscale(yscale)
    plt.legend()
    if not plot:
        plt.close(fig)
    
    results = dict(gain_vals=gain_vals, voltage=voltage)
    if plot:
        results["figure"] = fig
    if fit:
        results["model"] = model
    
    return results


def resolution_simple(
        context: dict,
        label: str = "",
        plot: bool = True,
        target: str | None = None
        ):
    """
    """
    results = context.get("results", {})
    if not results:
        return None
    if target not in results:
        line_vals = np.array([results[key]["line_val"] for key in results.keys()])
        sigma = np.array([results[key]["sigma"] for key in results.keys()])
        voltage = np.array([results[key]["voltage"] for key in results.keys()])
    else:
        line_vals = results[target]["line_val"]
        sigma = results[target]["sigma"]
        voltage = results[target]["voltage"]
        plot = False  # No plotting for single values
    res_vals = energy_resolution(line_vals, sigma)
    y = unumpy.nominal_values(res_vals)
    yerr = unumpy.std_devs(res_vals)
    fig = plt.figure("Energy Resolution vs Line Value")
    plt.errorbar(line_vals, y, yerr=yerr, fmt=".k", label=label)
    plt.xlabel("Voltage [V]")
    plt.ylabel(r"$\Delta$E / E")
    plt.legend()
    if not plot:
        plt.close(fig)

    results = dict(resolution_vals=res_vals, line_vals=line_vals)
    if plot:
        results["figure"] = fig
    return results


