"""Analysis tasks.
"""
from pathlib import Path

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import last_line_color, plt
from uncertainties import unumpy

from .config import (
    CalibrationDefaults,
    DriftDefaults,
    FitPeakDefaults,
    GainDefaults,
    PlotDefaults,
    ResolutionDefaults,
)
from .plotting import get_label, get_xrange, write_legend
from .utils import (
    SIGMA_TO_FWHM,
    amptek_accumulate_time,
    energy_resolution,
    energy_resolution_escape,
    find_peaks_iterative,
    gain,
    load_class,
)


def calibration(
          context: dict,
          charge_conversion: bool = CalibrationDefaults.charge_conversion,
          plot: bool = CalibrationDefaults.plot
      ) -> dict:
    """Perform the calibration of the detector using pulse data at fixed voltages.

    Arguments
    ---------
    context : dict
        The context dictionary containing the pulse data in `context["calibration"]` as an instance
        of the class PulsatorFile.
    charge_conversion : bool, optional
        Whether to convert the calibration to charge (fC) or leave it in voltage (mV).
        Default is True.
    plot : bool, optional
        Whether to generate and show the plots of the calibration process. Default is True.
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the calibration results in
        `context["calibration"]`.
    """
    # Get the histogram of the data and plot it
    pulse = context["calibration"]["pulse"]
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
    # Plot the calibration results
    cal_fig = plt.figure("Calibration")
    plt.errorbar(xdata, ydata, fmt=".k", label="Data")
    model.plot(fit_output=True, color=last_line_color())
    plt.legend()
    if not plot:
        plt.close(pulse_fig)
        plt.close(cal_fig)
    # Update the context with the calibration model
    context["calibration"]["model"] = model
    return context


def fit_peak(
          context: dict,
          subtask: str | None = None,
          model_class: AbstractFitModel = FitPeakDefaults.model_class,
          xmin: float = FitPeakDefaults.xmin,
          xmax: float = FitPeakDefaults.xmax,
          num_sigma_left: float = FitPeakDefaults.num_sigma_left,
          num_sigma_right: float = FitPeakDefaults.num_sigma_right,
          absolute_sigma: bool = FitPeakDefaults.absolute_sigma,
          p0: list[float] | None = FitPeakDefaults.p0
      ) -> dict:
    """Perform the fitting of a spectral emission line in the source data.

    Arguments
    ---------
    context : dict
        The context dictionary containing the source data in `context["tmp_source"]` as an instance
        of the class SourceFile.
    subtask: str
        The name of the fitting subtask.
    model_class : AbstractFitModel, optional
        The class of the model to use for fitting the spectral line. Default is Gaussian.
    xmin : float, optional
        The minimum x value to consider for the fit range. Default is -inf.
    xmax : float, optional
        The maximum x value to consider for the fit range. Default is inf.
    num_sigma_left : float, optional
        The number of sigmas to extend the fit range to the left of the peak. Default is 1.5.
    num_sigma_right : float, optional
        The number of sigmas to extend the fit range to the right of the peak. Default is 1.5.
    absolute_sigma : bool, optional
        Whether to use absolute sigma values for the fit. Default is True.
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the fit results in `context["results"]`.
    """
    # Access the source data from the context and get the histogram
    source = context["tmp_source"]
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
         absolute_sigma=absolute_sigma,
         p0=p0)
    # Initialize the model and fit the data.
    # model_class is given as a list of models, even if it contains only one model, but so far
    # we only support fitting a single model at a time in this context.
    model = model_class[0]()
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
    # Update the context with the fit results
    subtask_results = dict(line_val=line_val, sigma=sigma, voltage=source.voltage, model=model)
    file_name = source.file_path.stem
    if file_name not in context["fit"]:
        context["fit"][file_name] = {}
    context["fit"][file_name]["source"] = source
    context["fit"][file_name][subtask] = subtask_results
    return context


def gain_single(
        context: dict,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        target: str | None = None,
        **kwargs
        ):
    """Calculate the gain of the detector using the fit results obtained from the source data.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    w : float, optional
        The W-value of the gas inside the detector. Default is 26.0 eV (Ar).
    energy : float, optional
        The energy of the emission line used for gain calculation. Default is 5.895 keV (Fe-55 Kα).
    target : str, optional
        The name of the fitting subtask to use for gain calculation. If None, no calculation is
        performed. Default is None.
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the gain results in `context["results"]`.
    """
    task = "gain"
    fit_results = context.get("fit", {})
    file_name, = fit_results.keys()
    # Check if the target fitting subtask exists in the results and get the line position and
    # back voltage
    if target not in fit_results[file_name]:
        return context
    target_context = fit_results[file_name][target]
    line_val = target_context["line_val"]
    voltage = target_context["voltage"]
    # Calculate the gain and update the context
    gain_val = gain(w, line_val, energy)
    target_context[task] = gain_val
    # Create a label for the gain value to show if task is plotted
    target_context[f"{task}_label"] = f"Gain@{voltage:.0f} V: {gain_val}"
    if file_name not in context["results"]:
        context["results"][file_name] = {}
    context["results"][file_name][target] = target_context
    return context


def gain_folder(
        context: dict,
        target: str | None = None,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        fit: bool = GainDefaults.fit,
        plot: bool =  GainDefaults.plot,
        label: str | None = GainDefaults.label,
        yscale: str = GainDefaults.yscale
        ) -> dict:
    """Calculate the gain of the detector vs the back voltage using the fit results obtained from
    the source data of multiple files.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    target : str, optional
        The name of the fitting subtask to use for gain calculation. If None, no calculation is
        performed. Default is None.
    w : float, optional
        The W-value of the gas inside the detector. Default is 26.0 eV (Ar).
    energy : float, optional
        The energy of the emission line used for gain calculation. Default is 5.895 keV (Fe-55 Kα).
    fit : bool, optional
        Whether to fit the gain trend with an exponential model. Default is True.
    plot : bool, optional
        Whether to show the plots of the gain trend. Default is True.
    label : str, optional
        The label for the gain trend plot. Default is None.
    yscale : str, optional
        The y-axis scale for the gain trend plot. Can be "linear" or "log". Default is "log".
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the gain results in `context["results"]`.
    """
    task = "gain"
    fit_results = context.get("fit", {})
    # Get the different file names and create arrays to store gain values and voltages
    file_names = list(fit_results.keys())
    gain_vals = np.zeros(len(file_names), dtype=object)
    voltages = np.zeros(len(file_names))
    # Iterate over all files and calculate the gain values
    for i, file_name in enumerate(file_names):
        target_context = fit_results[file_name][target]
        line_val = target_context["line_val"]
        gain_vals[i] = gain(w, line_val, energy)
        voltages[i] = target_context["voltage"]
        task_label = f"Gain@{voltages[i]:.0f} V: {gain_vals[i]}"
        target_context[f"{task}_label"] = task_label
    y = unumpy.nominal_values(gain_vals)
    yerr = unumpy.std_devs(gain_vals)
    # Create the figure for the gain trend
    fig = plt.figure("Gain vs Voltage")
    plt.errorbar(voltages, y, yerr=yerr, fmt=".", label="Gain")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.yscale(yscale)
    # If fit is requested, fit the gain trend with an exponential model
    if fit:
        model = aptapy.models.Exponential()
        model.fit(voltages, y, sigma=yerr, absolute_sigma=True)
        model.plot(label=f"Scale: {-model.scale.ufloat()} V", color=last_line_color())
    write_legend(label)
    if not plot:
        plt.close(fig)
    # Update the context with the gain trend results and fit model
    context["results"][task] = dict(voltages=voltages, gain_vals=gain_vals)
    if fit:
        context["results"][task]["model"] = model
    return context


def gain_trend(
        context: dict,
        target: str | None = None,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        subtasks: list[str] | None = None,
    ) -> dict:
    """Calculate the gain of the detector vs time using the fit results obtained from the source
    data.
    
    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    target : str, optional
        The name of the fitting subtask to use for gain calculation. If None, no calculation is
        performed. Default is None.
    w : float, optional
        The W-value of the gas inside the detector. Default is 26.0 eV (Ar).
    energy : float, optional
        The energy of the emission line used for gain calculation. Default is 5.895 keV (Fe-55 Kα).
    subtasks : list[str], optional
        The list of fitting subtasks to fit the gain trend. If None, no fitting is performed.
        Default is None.
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the gain trend and fit results in
        `context["results"]`.
    """
    task = "gain_trend"
    fit_results = context.get("fit", {})
    # Get the different file names and create arrays to store gain values and times
    file_names = list(fit_results.keys())
    gain_vals = np.zeros(len(file_names), dtype=object)
    start_times = np.zeros(len(file_names), dtype=object)
    real_times = np.zeros(len(file_names))
    # Iterate over all files and calculate the gain values
    for i, file_name in enumerate(file_names):
        target_context = fit_results[file_name][target]
        source = fit_results[file_name]["source"]
        line_val = target_context["line_val"]
        gain_vals[i] =  gain(w, line_val, energy)
        start_times[i] = source.start_time
        real_times[i] = source.real_time
    times = amptek_accumulate_time(start_times, real_times) / 3600
    y = unumpy.nominal_values(gain_vals)
    yerr = unumpy.std_devs(gain_vals)
    # Save the gain trend values
    context["results"][task][target] = dict(times=times, gain_vals=gain_vals)
    plt.figure()
    plt.errorbar(times, y, yerr=yerr, fmt=".", label="Gain")
    # If fitting subtasks are provided, fit the gain trend with the specified models
    if subtasks is not None:
        for subtask in subtasks:
            # Think how to refactor this part
            model_list = load_class(subtask["model"])
            model = model_list[0]()
            for m in model_list[1:]:
                model += m()
            fit_pars = subtask.get("fit_pars", {})
            kwargs = dict(
                xmin=fit_pars["xmin"],
                xmax=fit_pars["xmax"],
                absolute_sigma=fit_pars["absolute_sigma"],
                p0=fit_pars["p0"]
            )
            model.fit(times, y, sigma=yerr, **kwargs)
            model.plot(fit_output=True, plot_components=False)
            # Update the context with the fit results
            context["results"][task][target][subtask["subtask"]] = dict(model=model)
    plt.legend()
    plt.show()
    return context


def compare_gain(
        context: dict,
        aggregate: bool = False,
        label: str | None = None,
        yscale: str = "log"
        ) -> dict:
    """Compare the gain of multiple folders vs voltage using the fit results obtained from the
    source data.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    aggregate : bool, optional
        Whether to aggregate all gain data from different folders and fit them together. Default is
        False.
    label : str, optional
        The label for the gain comparison plot. Default is None.
    yscale : str, optional
        The y-axis scale for the gain comparison plot. Can be "linear" or "log". Default is "log".
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the gain comparison results in
        `context["results"]`.
    """
    task = "compare_gain"
    folders = context.get("folders", {})
    plt.figure("gain_comparison")
    y = []
    yerr = []
    x = []
    for folder_path, folder_context in folders.items():
        folder_results = folder_context.get("results", {})
        folder_gain = folder_results.get("gain", {})
        g_val = unumpy.nominal_values(folder_gain.get("gain_vals", []))
        g_err = unumpy.std_devs(folder_gain.get("gain_vals", []))
        voltages = folder_gain.get("voltages", [])
        if not aggregate:
            model = folder_gain.get("model", None)
            plt.errorbar(voltages, g_val, yerr=g_err, fmt=".", label=Path(folder_path).stem)
            model.plot(label=f"Scale: {-model.scale.ufloat()} V", color=last_line_color())
        else:
            y.append(g_val)
            yerr.append(g_err)
            x.append(voltages)
    if aggregate:
        y = np.array(y).flatten()
        yerr = np.array(yerr).flatten()
        x = np.array(x).flatten()
        model = aptapy.models.Exponential()
        model.fit(x, y, sigma=yerr, absolute_sigma=True)
        plt.errorbar(x, y, yerr=yerr, fmt=".")
        model.plot(label=f"Scale: {-model.scale.ufloat()} V", color=last_line_color())
        context["results"][task] = dict(model=model)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.yscale(yscale)
    write_legend(label)
    return context


def resolution_single(
        context: dict,
        target: str | None = None,
        **kwargs
        ) -> dict:
    """Calculate the energy resolution of the detector using the fit results obtained from the
    source data. This calculation is based on the position and the width of the target spectral
    line.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    target : str, optional
        The name of the fitting subtask to use for resolution calculation. If None, no calculation
        is performed. Default is None.

    Returns
    -------
    context : dict
        The updated context dictionary containing the resolution results in `context["results"]`.
    """
    task = "resolution"
    fit_results = context.get("fit", {})
    file_name, = fit_results.keys()
    # Check if the target fitting subtask exists in the results and get the line position and sigma
    # of the target spectral line
    if target not in fit_results[file_name]:
        return context
    target_context = fit_results[file_name][target]
    line_vals = target_context["line_val"]
    sigma = target_context["sigma"]
    # Calculate the energy resolution and update the context
    res_val = energy_resolution(line_vals, sigma)
    fwhm = SIGMA_TO_FWHM * sigma
    target_context[task] = res_val
    # Get the energy of the emission line from the source configuration to create a label to show
    # if task is plotted
    energy = context["config"].source.e_peak
    task_label = f"FWHM@{energy:.1f} keV: {fwhm}\n" + fr"$\Delta$E/E: {res_val} %"
    target_context[f"{task}_label"] = task_label
    context["fit"][file_name][target] = target_context
    return context


def resolution_escape(
        context: dict,
        target_main: str | None = None,
        target_escape: str | None = None,
        **kwargs
        ):
    """Calculate the energy resolution of the detector using the fit results obtained from the
    source data. This calculation is based on the position and width of the main spectral line and
    the position of the escape peak.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    target_main : str, optional
        The name of the fitting subtask corresponding to the main spectral line. If None, no
        calculation is performed. Default is None.
    target_escape : str, optional
        The name of the fitting subtask corresponding to the escape peak. If None, no calculation is
        performed. Default is None.
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the resolution results in `context["results"]`.
    """
    task = "resolution_escape"
    fit_results = context.get("fit", {})
    file_name, = fit_results.keys()
    # Check if the main peak and escape peak fitting substasks exist in the results and get the
    # line positions and sigma of the main peak
    if target_main not in fit_results[file_name] or target_escape not in fit_results[file_name]:
        return context
    target_context = fit_results[file_name][target_main]
    line_main = target_context["line_val"]
    sigma_main = target_context["sigma"]
    line_escape = fit_results[file_name][target_escape]["line_val"]
    # Calculate the energy resolution using the escape peak and update the context
    res_val = energy_resolution_escape(line_main, line_escape, sigma_main)
    target_context[task] = res_val
    # Create a label for the resolution value to show if task is plotted
    target_context[f"{task}_label"] = fr"$\Delta$E/E(esc.): {res_val} %"
    context["results"][file_name][target_main] = target_context
    return context


def resolution_folder(
        context: dict,
        target: str | None = None,
        plot: bool = ResolutionDefaults.plot,
        label: str | None = ResolutionDefaults.label
        ) -> dict:
    """Calculate the energy resolution of the detector using the fit results obtained from the
    source data of multiple files.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    target : str, optional
        The name of the fitting subtask to use for resolution calculation. If None, no calculation
        is performed. Default is None.
    plot : bool, optional
        Whether to show the plots of the resolution trend. Default is True.
    label : str, optional
        The label for the resolution trend plot. Default is None.
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the resolution results in `context["results"]`.
    """
    task = "resolution"
    fit_results = context.get("fit", {})
    # Get the different file names and create arrays to store resolution values and voltages
    file_names = list(fit_results.keys())
    res_vals = np.zeros(len(file_names), dtype=object)
    voltages = np.zeros(len(file_names))
    # Iterate over all files and calculate the resolution values
    for i, file_name in enumerate(file_names):
        target_context = fit_results[file_name][target]
        sigma = target_context["sigma"]
        line_val = target_context["line_val"]
        res_vals[i] = energy_resolution(line_val, sigma)
        voltages[i] = target_context["voltage"]
        energy = context["config"].source.e_peak
        fwhm = SIGMA_TO_FWHM * sigma
        task_label = f"FWHM@{energy:.1f} keV: {fwhm}\n" + fr"$\Delta$E/E: {res_vals[i]} %"
        target_context[f"{task}_label"] = task_label
    y = unumpy.nominal_values(res_vals)
    yerr = unumpy.std_devs(res_vals)
    min_idx = np.argmin(y)
    # Create the figure for the resolution trend
    fig = plt.figure("Resolution vs Voltage")
    plt.errorbar(voltages, y, yerr=yerr, fmt=".k", label=label)
    plt.annotate(f"{y[min_idx]:.2f}", xy=(voltages[min_idx], y[min_idx]), xytext=(0, 30),
                 textcoords="offset points", ha="center", va="top", fontsize=12)
    plt.xlabel("Voltage [V]")
    plt.ylabel(r"$\Delta$E/E")
    write_legend(label)
    if not plot:
        plt.close(fig)
    # Update the context with the resolution trend results
    context["results"][task] = dict(voltages=voltages, res_vals=res_vals)
    return context


def drift(
        context: dict,
        target: str | None = None,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        threshold: float = DriftDefaults.threshold,
        plot: bool = DriftDefaults.plot,
        rate: bool = DriftDefaults.rate,
        label: str | None = DriftDefaults.label,
        yscale: str | None = DriftDefaults.yscale,
        **kwargs
        ) -> dict:
    """Calculate the gain and rate of the detector vs the drift voltage using the fit results
    obtained from the source data of multiple files.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    target : str, optional
        The name of the fitting subtask to use for gain calculation. If None, no calculation is
        performed. Default is None.
    w : float, optional
        The W-value of the gas inside the detector. Default is 26.0 eV (Ar).
    energy : float, optional
        The energy of the emission line used for gain calculation. Default is 5.895 keV (Fe-55 Kα).
    threshold : float, optional
        The energy threshold (in keV) above which to calculate the rate. Default is 1.5 keV.
    plot : bool, optional
        Whether to show the plots of the gain vs drift voltage. Default is True.
    rate : bool, optional
        Whether to plot the rate on a secondary y-axis. Default is False.
    label : str, optional
        The label for the gain trend plot. Default is None.
    yscale : str, optional
        The y-axis scale for the gain trend plot. Can be "linear" or "log". Default is "linear".
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the drift results in `context["results"]`.
    """
    task = "drift"
    # Get the different file names and create arrays to store rate values and drift voltages
    fit_results = context.get("fit", {})
    file_names = fit_results.keys()
    rates = np.zeros(len(fit_results), dtype=object)
    drift_voltages = np.zeros(len(fit_results))
    gain_vals = np.zeros(len(fit_results), dtype=object)
    # Iterate over all files and calculate the rate values
    for i, file_name in enumerate(file_names):
        source = fit_results[file_name]["source"]
        target_context = fit_results[file_name][target]
        line_val = target_context["line_val"]
        integration_time = source.real_time
        gain_vals[i] = gain(w, line_val, energy)
        drift_voltages[i] = source.drift_voltage
        # Calculate the threshold in charge and calculate the rate
        charge_thr = (threshold / energy) * line_val
        hist = source.hist
        counts = hist.content[hist.bin_centers() > charge_thr.n].sum()
        rates[i] = counts / integration_time
    # Prepare the quantities for plotting
    y = unumpy.nominal_values(gain_vals)
    yerr = unumpy.std_devs(gain_vals)
    y_rate = unumpy.nominal_values(rates)
    yerr_rate = unumpy.std_devs(rates)
    # Plot the gain vs drift voltage
    fig, ax1 = plt.subplots(num="drift_voltage")
    ax1.errorbar(drift_voltages, y, yerr=yerr, fmt=".k", label="Gain")
    ax1.set_xlabel("Drift Voltage [V]")
    ax1.set_ylabel("Gain", color=last_line_color())
    ax1.tick_params(axis="y", labelcolor=last_line_color())
    ax1.set_yscale(yscale)
    # Plot the rate on a secondary y-axis if requested
    if rate:
        color = "red"
        ax2 = ax1.twinx()
        ax2.errorbar(drift_voltages, y_rate, yerr=yerr_rate, fmt=".", color=color, label="Rate")
        ax2.set_ylabel("Rate [counts/s]", color=color)
        ax2.tick_params(axis="y",labelcolor=color)
    axs = (ax1, ax2) if rate else (ax1, )
    write_legend(label, *axs, loc="lower right")
    if not plot:
        plt.close(fig)
    # Update the context with the drift results
    context["results"][task] = dict(drift_voltages=drift_voltages,
                                    gain_vals=gain_vals,
                                    rates=rates)
    return context


def plot_spectrum(
        context: dict,
        targets: list[str] | None = None,
        xrange: list[float] | None = PlotDefaults.xrange,
        label: str | None = PlotDefaults.label,
        task_labels: list[str] | None = PlotDefaults.task_labels,
        loc: str = PlotDefaults.loc
        ) -> dict:
    """Plot the spectra from the source data and overlay the fitted models for the specified
    targets.
    
    Arguments
    ---------
    context : dict
        The context dictionary containing the source data in `context["sources"]` and the fit
        results in `context["fit"]`.
    targets : list[str], optional
        The list of fitting subtask names to plot the fitted models for. If None, no models are
        plotted. Default is None.
    xrange : list[float], optional
        The x-axis range for the plot. If None, the range is automatically calculated based on
        the data and fitted models. Default is None.
    label : str, optional
        The label for the plot legend. Default is None.
    task_labels : list[str], optional
        The list of task names to use for generating the labels of the fitted models. If None,
        no labels are generated. Default is None.
    loc : str, optional
        The location of the legend in the plot. Default is "best".
    
    Returns
    -------
    context : dict
        The context dictionary (in future it will be updated with the figures).
    """
    # Access the folder fit results from the context
    sources = context.get("sources", {})
    file_names = sources.keys()
    for file_name in file_names:
        source = sources[file_name]
        # Create the plot figure and plot the spectrum
        plt.figure(f"{source.file_path.stem}_{targets}")
        source.hist.plot(label="Data")
        # Plot the fitted models for the specified targets and get labels
        fit_results = context.get("fit", {})
        models = []
        if fit_results and targets is not None:
            for target in targets:
                if target in fit_results[file_name]:
                    target_context = fit_results[file_name][target]
                    model = target_context["model"]
                    fit_label = get_label(task_labels, target_context)
                    # Save the model for automatic xrange calculation
                    models.append(model)
                    model.plot(label=fit_label)
        # Set the x-axis range
        plt.xlim(xrange)
        if xrange is None:
            plt.xlim(get_xrange(source, models))
        write_legend(label, loc=loc)
    return context
