"""Analysis tasks.
"""
import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import last_line_color, plt
from uncertainties import unumpy

from .config import (
    CalibrationDefaults,
    FitPeakDefaults,
    GainDefaults,
    PlotDefaults,
    ResolutionDefaults)
from .fileio import SourceFile
from .utils import (
    SIGMA_TO_FWHM,
    energy_resolution,
    energy_resolution_escape,
    find_peaks_iterative,
    gain,
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
        The context dictionary containing the pulse data in `context["pulse"]` as an instance of
        the class PulsatorFile.
    charge_conversion : bool, optional
        Whether to convert the calibration to charge (fC) or leave it in voltage (mV).
        Default is True.
    plot : bool, optional
        Whether to generate and show the plots of the calibration process. Default is True.
    
    Returns
    -------
    context : dict
        The updated context dictionary containing the calibration results in `context["results"]`.
    """
    # Get the histogram of the data and plot it
    pulse = context["pulse"]
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
    # Update the context with the calibration results
    context["calibration"] = dict(model=model,
                                  pulse_figure=pulse_fig,
                                  calibration_figure=cal_fig)
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
    """Calculate the gain of the detector using the fit results obtained from the source data of
    multiple files.

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
    y = unumpy.nominal_values(gain_vals)
    yerr = unumpy.std_devs(gain_vals)
    # Create the figure for the gain trend
    fig = plt.figure("Gain vs Voltage")
    plt.errorbar(voltages, y, yerr=yerr, fmt=".", label=label)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.yscale(yscale)
    # If fit is requested, fit the gain trend with an exponential model
    if fit:
        model = aptapy.models.Exponential()
        model.fit(voltages, y, sigma=yerr, absolute_sigma=True)
        model.plot(label=f"Scale: {-model.scale.ufloat()} V", color=last_line_color())
    plt.legend()
    if not plot:
        plt.close(fig)
    # Update the context with the gain trend results and fit model
    context["results"][task] = dict(voltages=voltages, gain_vals=gain_vals)
    if fit:
        context["results"][task]["model"] = model
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
    context["results"][file_name][target] = target_context
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
    """Calculate the energy resolution of the detector using the fit results obtained from the source
    data of multiple files.

    Arguments
    ---------
    context : dict
        The context dictionary containing the fit results in `context["fit"]`.
    target : str, optional
        The name of the fitting subtask to use for resolution calculation. If None, no calculation is
        performed. Default is None.
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
    if not plot:
        plt.close(fig)
    # Update the context with the resolution trend results
    context["results"][task] = dict(voltages=voltages, res_vals=res_vals)
    # I have to find a way to save the label for each file
    return context


def drift_rate(
        context: dict,
        target: str | None = None,
        energy: float = 5.9,
        threshold: float = 1.5,
        plot: bool = True,
        label: str | None = None,
        **kwargs
        ) -> dict:
    task = "rate"
    fit_results = context.get("fit", {})
    file_names = fit_results.keys()
    rates = np.zeros(len(fit_results), dtype=object)
    drift_voltages = np.zeros(len(fit_results))
    for i, file_name in enumerate(file_names):
        source = fit_results[file_name]["source"]
        target_context = fit_results[file_name][target]
        line_val = target_context["line_val"]
        integration_time = source.real_time
        drift_voltages[i] = source.drift_voltage
        charge_thr = (threshold / energy) * line_val
        hist = source.hist
        area = hist.content * hist.bin_widths()
        counts = area[hist.bin_centers() > charge_thr.n].sum()
        rates[i] = counts / integration_time
    y = unumpy.nominal_values(rates)
    yerr = unumpy.std_devs(rates)
    fig = plt.figure("Rate vs Drift Voltage")
    plt.errorbar(drift_voltages, y, yerr=yerr, fmt=".", label=label)
    plt.xlabel("Drift Voltage [V]")
    plt.ylabel("Rate [counts/s]")

    return context



        


def _get_label(
        task_labels: list[str],
        target_context: dict
        ) -> str | None:
    """Generate a label for the plot based on the specified task labels and the target context.

    Arguments
    ---------
    task_labels : list[str]
        The list of task names whose labels should be included in the plot legend.
    target_context : dict
        The context dictionary containing the labels for each task.

    Returns
    -------
    label : str | None
        The generated label for the plot, or None if no task labels are provided.
    """
    # Check if task_labels is provided
    if task_labels is None:
        return None
    label = ""
    # Iterate over the task labels and append the corresponding labels from the target context
    for task in task_labels:
        if task in target_context:
            task_label = target_context[f"{task}_label"]
            label += f"{task_label}\n"
    # Remove the trailing newline character
    label = label[:-1]
    return label


def _get_xrange(source: SourceFile, models: list[AbstractFitModel]) -> list[float]:
    """Determine the x-axis range for plotting the source spectrum and fitted models.
    
    Arguments
    ---------
    source : SourceFile
        The source data file containing the histogram to plot.
    models : list[AbstractFitModel]
        The list of fitted models to consider for determining the x-axis range.
    Returns
    -------
    xrange : list[float]
        The x-axis range `[xmin, xmax]` for plotting.
    """
    # Get the histogram range with counts > 0 and get the range
    content = np.insert(source.hist.content, -1, 0)
    edges = source.hist.bin_edges()[content > 0]
    xmin, xmax = edges[0], edges[-1]
    m_mins = []
    m_maxs = []
    # Get the plotting range of all models, if any
    for model in models:
        low, high = model.default_plotting_range()
        m_mins.append(low)
        m_maxs.append(high)
    # Determine the final xrange including all models. If no model is given, use the
    # histogram range
    if len(models) > 0:
        xmin = max(xmin, min(m_mins))
        xmax = min(xmax, max(m_maxs))
    return [xmin, xmax]


def plot_spec(
        context: dict,
        targets: str | None = None,
        label: str | None = PlotDefaults.label,
        xrange: list[float] | None = PlotDefaults.xrange,
        task_labels: list[str] | None = PlotDefaults.task_labels
        ) -> None:
    """Plot the source spectrum along with the fitted models for the specified targets.

    Arguments
    ---------
    context : dict
        The context dictionary containing the source data in `context["source"]` and the fit
        results in `context["results"]`.
    targets : list[str], optional
        The list of fitting subtask names to plot. If None, the spectrum histogram is plotted with
        no fitted models. Default is None.
    label : str, optional
        The label for the source histogram. Default is "".
    xrange : list[float], optional
        The x-axis range for the plot. If None, the range is determined automatically to include
        all the target fitted models. Default is None.
    task_labels : list[str], optional
        The list of task names whose labels should be included in the plot legend. If None, no
        labels are included. Default is None.

    Returns
    -------
    context : dict
        The unchanged context dictionary.
    """
    # Access the source data and results from the context
    source = context.get("tmp_source")
    results = context.get("results", {})
    file_name, = results.keys()
    # Create the plot figure and plot the spectrum
    plt.figure(f"{source.file_path.stem}_{targets} ")
    source.hist.plot(label=label)
    # Plot the fitted models for the specified targets and get labels
    models = []
    if targets is not None:
        for target in targets:
            if target in results[file_name]:
                target_context = results[file_name][target]
                model = target_context["model"]
                label = _get_label(task_labels, target_context)
                # Save the model for automatic xrange calculation
                models.append(model)
                model.plot(label=label)
    # Set the x-axis range
    if xrange is None:
        xrange = _get_xrange(source, models)
    plt.xlim(xrange)
    plt.legend()
    return context
