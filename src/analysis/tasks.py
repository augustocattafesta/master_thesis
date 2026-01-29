"""Analysis tasks.
"""
from typing import Any, Literal

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
from .context import Context, FoldersContext, TargetContext
from .plotting import get_label, get_xrange, write_legend
from .utils import (
    amptek_accumulate_time,
    energy_resolution,
    energy_resolution_escape,
    find_peaks_iterative,
    gain,
    load_class,
)


def calibration(
          context: Context,
          charge_conversion: bool = CalibrationDefaults.charge_conversion,
          plot: bool = CalibrationDefaults.plot
      ) -> Context:
    """Perform the calibration of the detector using pulse data at fixed voltages.

    Parameters
    ---------
    context : Context
        The context object containing the pulse data in `context.pulse` as an instance
        of the class PulsatorFile.
    charge_conversion : bool, optional
        Whether to convert the calibration to charge (fC) or leave it in voltage (mV).
        Default is True.
    plot : bool, optional
        Whether to generate and show the plots of the calibration process. Default is True.
    
    Returns
    -------
    context : Context
        The updated context object containing the calibration results in
        `context.conversion_model`.
    """
    # Get the histogram of the data and plot it
    pulse = context.pulse
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
    context.conversion_model = model
    # Update the context with the figures
    context.add_figure("pulse", pulse_fig)
    context.add_figure("calibration", cal_fig)
    return context


def fit_peak(
          context: Context,
          target: str,
          model_class: list[type[AbstractFitModel]],
          xmin: float = FitPeakDefaults.xmin,
          xmax: float = FitPeakDefaults.xmax,
          num_sigma_left: float = FitPeakDefaults.num_sigma_left,
          num_sigma_right: float = FitPeakDefaults.num_sigma_right,
          absolute_sigma: bool = FitPeakDefaults.absolute_sigma,
          p0: list[float] | None = FitPeakDefaults.p0
      ) -> Context:
    """Perform the fitting of a spectral emission line in the source data.

    Parameters
    ---------
    context : Context
        The context object containing the source data in `context.last_source` as an instance
        of the class SourceFile.
    target: str
        The name of the fitting target.
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
    context : Context
        The updated context object containing the fit results.
    """
    # Access the last source data added to the context and get the histogram
    source = context.last_source
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
        model.intensity1.freeze(0.16)   # type: ignore[attr-defined]
    model.fit_iterative(hist, **kwargs) # type: ignore[attr-defined]
    # Extract the line value and sigma from the fit results
    if isinstance(model, aptapy.models.Gaussian):
        line_val = model.status.correlated_pars[1]
        sigma = model.status.correlated_pars[2]
    elif isinstance(model, aptapy.models.Fe55Forest):
        reference_energy: float = model.energies[0]   # type: ignore [attr-defined]
        line_val = reference_energy / model.status.correlated_pars[1]
        sigma = model.status.correlated_pars[2]
    else:
        raise TypeError(f"Model of type {type(model)} not supported in fit_peak task")
    # Update the context with the fit results
    target_ctx = TargetContext(target, line_val, sigma, source.voltage, model)
    target_ctx.energy = context.config.acquisition.e_peak
    context.add_target_ctx(source, target_ctx)
    return context


def gain_task(
        context: Context,
        target: str,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        fit: bool = GainDefaults.fit,
        plot: bool = GainDefaults.plot,
        label: str | None = GainDefaults.label,
        yscale: str = GainDefaults.yscale
        ) -> Context:
    """Calculate the gain of the detector vs the back voltage using the fit results obtained from
    the source data of multiple files.

    Parameters
    ---------
    context : Context
        The context object containing the fit results.
    target : str
        The name of the fitting subtask to use for gain calculation.
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
    context : Context
        The updated context object containing the gain results.
    """
    # pylint: disable=invalid-unary-operand-type
    task = "gain"
    # Get the file names from the fit context keys
    file_names = context.file_names
    # Create empty arrays to store gain values and voltages
    gain_vals = np.zeros(len(file_names), dtype=object)
    voltages = np.zeros(len(file_names))
    # Iterate over all files and calculate the gain
    for i, file_name in enumerate(file_names):
        target_ctx = context.target_ctx(file_name, target)
        line_val = target_ctx.line_val
        voltages[i] = target_ctx.voltage
        target_ctx.gain_val = gain(w, line_val, energy)
        gain_vals[i] = target_ctx.gain_val
    # Save the results in the context
    context.add_task_results(task, target, dict(voltages=voltages, gain_vals=gain_vals))
    # If only a single file is analyzed, return the context without plotting or fitting
    if len(file_names) == 1:
        return context
    y = unumpy.nominal_values(gain_vals)
    yerr = unumpy.std_devs(gain_vals)
    # Create the figure for the gain trend
    fig = plt.figure("gain_vs_voltage")
    plt.errorbar(voltages, y, yerr=yerr, fmt=".", label="Data")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.yscale(yscale)
    # If fit is requested, fit the gain trend with an exponential model
    if fit:
        model = aptapy.models.Exponential()
        model.fit(voltages, y, sigma=yerr, absolute_sigma=True)
        model.plot(label=f"Scale: {-model.scale.ufloat()} V", color=last_line_color())
        # Add the fit model to the context
        context.add_task_fit_model(task, target, model)
    # Write the legend and show or close the plot
    write_legend(label)
    if not plot:
        plt.close(fig)
    # Add the figure to the context
    context.add_figure(task, fig)
    return context


def gain_trend(
        context: Context,
        target: str,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        subtasks: list[dict[str, Any]] | None = None,
        label: str | None = GainDefaults.label
    ) -> Context:
    """Calculate the gain of the detector vs time using the fit results obtained from the source
    data.
    
    Parameters
    ---------
    context : Context
        The context object containing the fit results.
    target : str
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
    context : Context
        The updated context object containing the gain trend and fit results.
    """
    task = "gain_trend"
    # Get the different file names
    file_names = context.file_names
    # Create empty arrays to store gain values and start times
    gain_vals = np.zeros(len(file_names), dtype=object)
    start_times = np.zeros(len(file_names), dtype=object)
    real_times = np.zeros(len(file_names))
    # Iterate over all files and calculate the gain values
    for i, file_name in enumerate(file_names):
        # Access the source data to extract the times
        source = context.source(file_name)
        start_times[i] = source.start_time
        real_times[i] = source.real_time
        # Access the target context and extract line value and voltage
        target_ctx = context.target_ctx(file_name, target)
        line_val = target_ctx.line_val
        target_ctx.gain_val = gain(w, line_val, energy)
        gain_vals[i] = target_ctx.gain_val
    # Calculate the accumulated time in hours
    times = amptek_accumulate_time(start_times, real_times) / 3600
    # Save the results in the context
    context.add_task_results(task, target, dict(times=times, gain_vals=gain_vals))
    y = unumpy.nominal_values(gain_vals)
    yerr = unumpy.std_devs(gain_vals)
    # Create the figure for the gain trend
    fig = plt.figure("gain_vs_time")
    plt.errorbar(times, y, yerr=yerr, fmt=".", label="Data")
    plt.xlabel("Time [hours]")
    plt.ylabel("Gain")
    # If fitting subtasks are provided, fit the gain trend with the specified models
    if subtasks:
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
            context.add_subtask_fit_model(task, target, subtask["target"], model)
            # context["results"][task][target][name] = dict(model=model)
    write_legend(label)
    context.add_figure(task, fig)
    return context


def compare_gain(
        context: FoldersContext,
        target: str,
        combine: bool = False,
        label: str | None = None,
        yscale: Literal["linear", "log"] = "linear"
        ) -> FoldersContext:
    """Compare the gain of multiple folders vs voltage using the fit results obtained from the
    source data.

    Parameters
    ---------
    context : FoldersContext
        The context object containing the fit results.
    combine : bool, optional
        Whether to combine all gain data from different folders and fit them together. Default is
        False.
    label : str, optional
        The label for the gain comparison plot. Default is None.
    yscale : str, optional
        The y-axis scale for the gain comparison plot. Can be "linear" or "log". Default is "log".
    
    Returns
    -------
    context : FoldersContext
        The updated context object containing the gain comparison results.
    """
    # pylint: disable=invalid-unary-operand-type
    task = "compare_gain"
    # Get the different folder names
    folder_names = context.folder_names
    # Create empty arrays to store gain values and voltages
    y = np.zeros(len(folder_names), dtype=object)
    yerr = np.zeros(len(folder_names), dtype=object)
    x = np.zeros(len(folder_names), dtype=object)
    # Create the figure for the gain comparison
    fig = plt.figure("gain_comparison")
    # Iterate over all folders and plot the gain values
    for i, folder_name in enumerate(folder_names):
        folder_ctx = context.folder_ctx(folder_name)
        folder_gain = folder_ctx.task_results("gain", target)
        g_val = unumpy.nominal_values(folder_gain.get("gain_vals", []))
        g_err = unumpy.std_devs(folder_gain.get("gain_vals", []))
        voltages = folder_gain.get("voltages", [])
        # If not aggregating, plot each folder separately
        if not combine:
            plt.errorbar(voltages, g_val, yerr=g_err, fmt=".", label=folder_name)
            model = folder_gain.get("model", None)
            if model:
                model.plot(label=f"Scale: {-model.scale.ufloat()} V", color=last_line_color())
        # If aggregating, store the data together for later fitting and plotting
        else:
            y[i] = g_val
            yerr[i] = g_err
            x[i] = voltages
    if combine:
        # Concatenate all data and fit with an exponential model
        y = np.concatenate(y)
        yerr = np.concatenate(yerr)
        x = np.concatenate(x)
        model = aptapy.models.Exponential()
        model.fit(x, y, sigma=yerr, absolute_sigma=True)
        # Plot the aggregated data and fit model
        plt.errorbar(x, y, yerr=yerr, fmt=".")
        model.plot(label=f"Scale: {-model.scale.ufloat()} V", color=last_line_color())
        context.add_task_results(task, target, dict(model=model))
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.yscale(yscale)
    # Write the legend and show the plot
    write_legend(label)
    # Add the figure to the context
    context.add_figure(task, fig)
    return context


def resolution_task(
        context: Context,
        target: str,
        plot: bool = ResolutionDefaults.plot,
        label: str | None = ResolutionDefaults.label
        ) -> Context:
    """Calculate the energy resolution of the detector using the fit results obtained from the
    source data. This estimate is based on the position and the width of the target spectral
    line.

    Parameters
    ---------
    context : Context
        The context object containing the fit results.
    target : str, optional
        The name of the fitting subtask to use for resolution calculation. If None, no calculation
        is performed. Default is None.
    plot : bool, optional
        Whether to show the plots of the resolution trend. Default is True.
    label : str, optional
        The label for the resolution trend plot. Default is None.
    
    Returns
    -------
    context : Context
        The updated context object containing the resolution results.
    """
    task = "resolution"
    # Get the file names from the fit context keys
    file_names = context.file_names
    # Create empty arrays to store resolution values and voltages
    res_vals = np.zeros(len(file_names), dtype=object)
    voltages = np.zeros(len(file_names))
    # Iterate over all files and calculate the gain
    for i, file_name in enumerate(file_names):
        # Access the target context and extract line value, sigma and voltage
        target_ctx = context.target_ctx(file_name, target)
        line_val = target_ctx.line_val
        sigma = target_ctx.sigma
        voltages[i] = target_ctx.voltage
        target_ctx.res_val = energy_resolution(line_val, sigma)
        res_vals[i] = target_ctx.res_val
    # Save the results in the context
    context.add_task_results(task, target, dict(voltages=voltages, res_vals=res_vals))
    # If only a single file is analyzed, return the context without plotting
    if len(file_names) == 1:
        return context
    y = unumpy.nominal_values(res_vals)
    yerr = unumpy.std_devs(res_vals)
    min_idx = np.argmin(y)
    # Create the figure for the resolution trend
    fig = plt.figure("Resolution vs Voltage")
    plt.errorbar(voltages, y, yerr=yerr, fmt=".", label="Data")
    # Write the minimum resolution value on the plot
    plt.annotate(f"{y[min_idx]:.2f}", xy=(voltages[min_idx], y[min_idx]), xytext=(0, 30),
                 textcoords="offset points", ha="center", va="top", fontsize=12)
    plt.xlabel("Voltage [V]")
    plt.ylabel(r"$\Delta$E/E")
    # Write the legend and show or close the plot
    write_legend(label)
    if not plot:
        plt.close(fig)
    # Add the figure to the context
    context.add_figure(task, fig)
    return context


def resolution_escape(
        context: Context,
        target_main: str,
        target_escape: str,
        label: str | None = ResolutionDefaults.label,
        plot: bool = ResolutionDefaults.plot
        ) -> Context:
    """Calculate the energy resolution of the detector using the fit results obtained from the
    source data. This calculation is based on the position and width of the main spectral line and
    the position of the escape peak.

    Parameters
    ---------
    context : Context
        The context object containing the fit results.
    target_main : str, optional
        The name of the fitting subtask corresponding to the main spectral line. If None, no
        calculation is performed. Default is None.
    target_escape : str, optional
        The name of the fitting subtask corresponding to the escape peak. If None, no calculation is
        performed. Default is None.
    
    Returns
    -------
    context : Context
        The updated context object containing the resolution results.
    """
    task = "resolution_escape"
    # Get the single file names from the fit context keys
    file_names = context.file_names
    # Create empty arrays to store resolution values and voltages
    res_vals = np.zeros(len(file_names), dtype=object)
    voltages = np.zeros(len(file_names))
    for i, file_name in enumerate(file_names):
        # Access the target contexts and extract line values and sigma
        target_ctx = context.target_ctx(file_name, target_main)
        line_val_main = target_ctx.line_val
        sigma_main = target_ctx.sigma
        voltages[i] = target_ctx.voltage
        line_val_esc = context.target_ctx(file_name, target_escape).line_val
        # Calculate the energy resolution using the escape peak and update the context
        target_ctx.res_escape_val = energy_resolution_escape(line_val_main,
                                                             line_val_esc,
                                                             sigma_main)
        res_vals[i] = target_ctx.res_escape_val
    # Save the results in the context
    context.add_task_results(task, target_main, dict(voltages=voltages, res_vals=res_vals))
    # If only a single file is analyzed, return the context without plotting
    if len(file_names) == 1:
        return context
    y = unumpy.nominal_values(res_vals)
    yerr = unumpy.std_devs(res_vals)
    min_idx = np.argmin(y)
    # Create the figure for the resolution trend
    fig = plt.figure("Resolution vs Voltage")
    plt.errorbar(voltages, y, yerr=yerr, fmt=".k", label=label)
    # Write the minimum resolution value on the plot
    plt.annotate(f"{y[min_idx]:.2f}", xy=(voltages[min_idx], y[min_idx]), xytext=(0, 30),
                 textcoords="offset points", ha="center", va="top", fontsize=12)
    plt.xlabel("Voltage [V]")
    plt.ylabel(r"$\Delta$E/E")
    # Write the legend and show or close the plot
    write_legend(label)
    if not plot:
        plt.close(fig)
    # Add the figure to the context
    context.add_figure(task, fig)
    return context


def drift(
        context: Context,
        target: str,
        w: float = GainDefaults.w,
        energy: float = GainDefaults.energy,
        threshold: float = DriftDefaults.threshold,
        plot: bool = DriftDefaults.plot,
        rate: bool = DriftDefaults.rate,
        label: str | None = DriftDefaults.label,
        yscale: str = DriftDefaults.yscale,
        ) -> Context:
    """Calculate the gain and rate of the detector vs the drift voltage using the fit results
    obtained from the source data of multiple files.

    Parameters
    ---------
    context : Context
        The context object containing the fit results.
    target : str
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
    context : Context
        The updated context object containing the drift results.
    """
    task = "drift"
    # Get the different file names and create arrays to store rate values and drift voltages
    file_names = context.file_names
    rates = np.zeros(len(file_names), dtype=object)
    drift_voltages = np.zeros(len(file_names))
    gain_vals = np.zeros(len(file_names), dtype=object)
    # Iterate over all files and calculate the rate values
    for i, file_name in enumerate(file_names):
        source = context.source(file_name)
        drift_voltages[i] = source.drift_voltage
        integration_time = source.real_time
        target_ctx = context.target_ctx(file_name, target)
        line_val = target_ctx.line_val
        gain_vals[i] = gain(w, line_val, energy)
        # Calculate the threshold in charge and calculate the rate
        charge_thr = (threshold / energy) * line_val
        hist = source.hist
        counts = hist.content[hist.bin_centers() > charge_thr.n].sum()
        rates[i] = counts / integration_time
    context.add_task_results(task, target, dict(drift_voltages=drift_voltages,
                                               gain_vals=gain_vals,
                                               rates=rates))
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
    # Add the figure to the context
    context.add_figure(task, fig)
    return context


def plot_spectrum(
        context: Context,
        targets: list[str] | None = None,
        xrange: list[float] | None = PlotDefaults.xrange,
        label: str | None = PlotDefaults.label,
        task_labels: list[str] | None = PlotDefaults.task_labels,
        loc: str = PlotDefaults.loc
        ) -> Context:
    """Plot the spectra from the source data and overlay the fitted models for the specified
    targets.
    
    Parameters
    ---------
    context : Context
        The context object containing the source data the fit results.
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
    context : Context
        The context object (in future it will be updated with the figures).
    """
    # Get the file names from the sources keys
    file_names = context.file_names
    # Iterate over all files and plot the spectra with fitted models, if desired
    for file_name in file_names:
        # Create the plot figure and plot the spectrum
        source = context.source(file_name)
        fig = plt.figure(f"{source.file_path.stem}_{targets}")
        source.hist.plot(label="Data")
        # Plot the fitted models for the specified targets and get labels
        models = []
        if targets is not None:
            for target in targets:
                target_ctx = context.target_ctx(file_name, target)
                model = target_ctx.model
                model_label = get_label(task_labels, target_ctx)
                # Save the model for automatic xrange calculation
                models.append(model)
                model.plot(label=model_label)
        # Set the x-axis range
        plt.xlim(xrange)
        if xrange is None:
            plt.xlim(get_xrange(source, models))
        write_legend(label, loc=loc)
        context.add_figure(file_name, fig)
    return context
