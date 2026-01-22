"""Analysis tasks.
"""
import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import last_line_color, plt
from uncertainties import unumpy

from .config import CalibrationDefaults, FitPeakDefaults, GainDefaults, PlotDefaults
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
    context["results"]["calibration"] = dict(model=model,
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
        The context dictionary containing the source data in `context["source"]` as an instance of
        the class SourceFile.
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
    source = context["source"]
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
    subtask_results = dict(line_val=line_val, sigma=sigma, voltage=source.voltage, model=model)
    if subtask is None:
        subtask = source.file_path.stem
    context["results"][subtask] = subtask_results
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
        The context dictionary containing the fit results in `context["results"]`.
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
    results = context.get("results", {})
    # Check if the target fitting subtask exists in the results and get the line position and
    # back voltage
    if target not in results:
        return context
    target_context = results[target]
    line_val = target_context["line_val"]
    voltage = target_context["voltage"]
    # Calculate the gain and update the context
    gain_val = gain(w, line_val, energy)
    target_context[task] = gain_val
    # Create a label for the gain value to show if task is plotted
    target_context[f"{task}_label"] = f"Gain@{voltage:.0f} V: {gain_val}"
    context["results"][target] = target_context
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
        The context dictionary containing the fit results in `context["results"]`.
    target : str, optional
        The name of the fitting subtask to use for resolution calculation. If None, no calculation
        is performed. Default is None.

    Returns
    -------
    context : dict
        The updated context dictionary containing the resolution results in `context["results"]`.
    """
    task = "resolution"
    results = context.get("results", {})
    # Check if the target fitting subtask exists in the results and get the line position and sigma
    # of the target spectral line
    if target not in results:
        return context
    target_context = results[target]
    line_vals = target_context["line_val"]
    sigma = target_context["sigma"]
    # Calculate the energy resolution and update the context
    res_val = energy_resolution(line_vals, sigma)
    fwhm = SIGMA_TO_FWHM * sigma
    target_context[task] = res_val
    # Get the energy of the emission line from the source configuration to create a label to show
    # if task is plotted
    energy = context["config"].source.e_peak
    task_label = f"FWHM@{energy:.1f} keV: {fwhm}\n" + fr"$\Delta$E/E: {res_val}"
    target_context[f"{task}_label"] = task_label
    context["results"][target] = target_context
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
        The context dictionary containing the fit results in `context["results"]`.
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
    results = context.get("results", {})
    # Check if the main peak and escape peak fitting substasks exist in the results and get the
    # line positions and sigma of the main peak
    if target_main not in results or target_escape not in results:
        return context
    target_context = results[target_main]
    line_main = target_context["line_val"]
    sigma_main = target_context["sigma"]
    line_escape = results[target_escape]["line_val"]
    # Calculate the energy resolution using the escape peak and update the context
    res_val = energy_resolution_escape(line_main, line_escape, sigma_main)
    target_context[task] = res_val
    # Create a label for the resolution value to show if task is plotted
    target_context[f"{task}_label"] = fr"$\Delta$E/E(esc.): {res_val}"
    context["results"][target_main] = target_context
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
    source = context.get("source")
    results = context.get("results", {})
    # Create the plot figure and plot the spectrum
    plt.figure(f"{source.file_path.name}_{targets} ")
    source.hist.plot(label=label)
    # Plot the fitted models for the specified targets and get labels
    models = []
    if targets is not None:
        for target in targets:
            if target in results:
                target_context = results[target]
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
