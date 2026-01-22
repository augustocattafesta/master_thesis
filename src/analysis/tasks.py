from dataclasses import dataclass

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import last_line_color, plt
from uncertainties import unumpy

from .fileio import PulsatorFile, SourceFile
from .utils import (
    KALPHA,
    SIGMA_TO_FWHM,
    find_peaks_iterative,
    gain,
    energy_resolution,
    energy_resolution_escape
)



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


def gain_single(
        context: dict,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        target: str | None = None,
        ):
    """
    """
    task = "gain"
    results = context.get("results", {})
    if target not in results:
        return context
    else:
        target_context = results[target]
        line_vals = target_context["line_val"]
        voltage = target_context["voltage"]
        # fit, plot = False, False  # No fitting or plotting for single values
    gain_val = gain(w, line_vals, energy)
    target_context[task] = gain_val
    target_context[f"{task}_label"] = f"Gain@{voltage:.0f} V: {gain_val}"
    context["results"][target] = target_context
    # y = unumpy.nominal_values(gain_vals)
    # yerr = unumpy.std_devs(gain_vals)
    # fig = plt.figure("Gain vs voltage")
    # if fit:
    #     model = aptapy.models.Exponential()
    #     model.fit(voltage, y, sigma=yerr, absolute_sigma=True)
    #     fit_label = f"Scale: {-model.scale.ufloat()} V"
    #     model.plot(label=fit_label, color=last_line_color())
    # if label is None:
    #     folder_name = None
    #     label = load_label(folder_name)
    # plt.errorbar(voltage, y, yerr=yerr, fmt=".k", label=label)
    # plt.xlabel("Voltage [V]")
    # plt.ylabel("Gain")
    # plt.yscale(yscale)
    # plt.legend()
    # if not plot:
    #     plt.close(fig)
    
    # results = dict(gain_vals=gain_vals, voltage=voltage)
    # if plot:
    #     results["figure"] = fig
    # if fit:
    #     results["model"] = model
    
    return context


def resolution_single(
        context: dict,
        target: str | None = None
        ) -> dict:
    """
    """
    task = "resolution"
    results = context.get("results", {})
    if target not in results:
        return context
    else:
        target_context = results[target]
        line_vals = target_context["line_val"]
        sigma = target_context["sigma"]
        # voltage = target_context["voltage"]
        # plot = False  # No plotting for single values
    energy = context["config"].source.e_peak
    res_val = energy_resolution(line_vals, sigma)
    fwhm = SIGMA_TO_FWHM * sigma
    target_context[task] = res_val
    task_label = f"FWHM@{energy:.1f} keV: {fwhm}\n" + fr"$\Delta$E/E: {res_val}"
    target_context[f"{task}_label"] = task_label
    # y = unumpy.nominal_values(res_vals)
    # yerr = unumpy.std_devs(res_vals)
    # fig = plt.figure("Energy Resolution vs Line Value")
    # plt.errorbar(voltage, y, yerr=yerr, fmt=".k", label=label)
    # plt.xlabel("Voltage [V]")
    # plt.ylabel(r"$\Delta$E / E")
    # plt.legend()
    # if not plot:
    #     plt.close(fig)

    context["results"][target] = target_context
    # results = dict(resolution_vals=res_vals, line_vals=line_vals)
    # if plot:
    #     results["figure"] = fig


    return context


def resolution_escape(
        context: dict,
        target_main: str | None = None,
        target_escape: str | None = None
        ):
    """
    """
    task = "resolution_escape"
    results = context.get("results", {})
    if target_main not in results or target_escape not in results:
        return context
    else:
        target_context = results[target_main]
        line_main = target_context["line_val"]
        sigma_main = target_context["sigma"]
        line_escape = results[target_escape]["line_val"]
    res_val = energy_resolution_escape(line_main, line_escape, sigma_main)
    target_context[task] = res_val
    target_context[f"{task}_label"] = fr"$\Delta$E/E(esc.): {res_val}"
    context["results"][target_main] = target_context
    return context


def plot_spec(
        context: dict,
        plot: bool = True,
        targets: str | None = None,
        label: str | None = None,
        xrange: list[float] | None = None,
        task_labels: list[str] | None = None
        ) -> None:
    """
    """
    source = context.get("source", None)
    results = context.get("results", {})
    fig = plt.figure(source.file_path.name)
    source.hist.plot(label=label)
    content = np.insert(source.hist.content, -1, 0)
    edges = source.hist.bin_edges()[content > 0]
    xmin, xmax = edges[0], edges[-1]
    if targets is not None:
        for target in targets:
            if target in results:
                target_context = results[target]
                model = target_context["model"]
                if task_labels is not None:
                    label = ""
                    for task in task_labels:
                        if task in target_context:
                            task_label = target_context[f"{task}_label"]
                            label += f"{task_label}\n"
                model.plot(label=label)
                model_xmin, model_xmax = model.default_plotting_range()
                if xmin == edges[0]:
                    xmin = model_xmin
                else:
                    xmin = min(xmin, model_xmin)
                if xmax == edges[-1]:
                    xmax = model_xmax
                else:    
                    xmax = max(xmax, model_xmax)
    if xmin < edges[0]:
        xmin = edges[0]
    if xmax > edges[-1]:
        xmax = edges[-1]
    if xrange is None:
        xrange = [xmin, xmax]
    try:
        plt.xlim(xrange)
    except ValueError:
        print("warning: xrange must be a list of two floats.")
        xrange = [xmin, xmax]
        plt.xlim(xrange)

    plt.legend()
    
    if not plot:
        plt.close(fig)


