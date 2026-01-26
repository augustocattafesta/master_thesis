"""Module to analyze data
"""

from collections.abc import Sequence
from pathlib import Path

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel, line_forest
from aptapy.plotting import last_line_color, plt
from uncertainties import unumpy

from . import ANALYSIS_DATA
from .fileio import DataFolder, PulsatorFile, SourceFile, load_label
from .log import LogYaml
from .utils import AR_ESCAPE, KALPHA, KBETA, energy_resolution, gain


@line_forest(KALPHA - AR_ESCAPE, KBETA - AR_ESCAPE)
class ArEscape(aptapy.models.GaussianForestBase):
    pass

def analyze_file(pulse_file: str | Path, source_file: str | Path,
                 models: Sequence[type[AbstractFitModel]], w: float, capacity: float,
                 e_peak: float, plot: bool = False, save: bool = False,
                 **kwargs) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Analyze a calibration pulses file to determine the calibration parameters of the readout
    circuit. If a source data file (spectrum) is given, the emission line(s) is fitted using the
    given model. If multiple models are given, the fit is done with each model. 

    Arguments
    ----------
    pulse_file : Union[str, Path]
        Path of the calibration pulses file.
    source_file : Union[str, Path]
        Path of the source file.
    models : Tuple[AbstractFitModel]
        Model(s) to fit the source emission line.
    W : float
        W-value of the detector gas.
    capacity : float
        Capacity of the capacitance of the readout circuit of the detector.
    e_peak : float
        Energy of the main emission line of the source (in keV).
    plot : bool, optional
        If True, plot the figures of the analysis, by default False.
    save : bool, optional
        If True, save a log file and plots of the analysis, by default False.
    **kwargs : dict
        Refer to `fileio.SourceFile.fit`.

    Returns
    -------
    Union[ArrayLike, Tuple[float, float]]
        If only the calibration pulses file is given, the fit parameters of the line model are
        returned. If also a source file is given, the results of the energy resolution and the
        gain are returned.
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    # Start the logging
    logyaml = LogYaml()
    if save:
        logyaml.start_logging()
    try:
        # Pulse file analysis and plotting
        pulse_data = PulsatorFile(Path(pulse_file))
        line_model, pulse_fig, line_fig = pulse_data.analyze_pulses()
        logyaml.add_pulse_results(pulse_data.file_path.name, line_model)
        if not plot:
            plt.close(pulse_fig)
            plt.close(line_fig)
        if save:
            pulse_fig.savefig(logyaml.log_folder / "cal_pulses.pdf", format="pdf")
            line_fig.savefig(logyaml.log_folder / "cal_fit.pdf", format="pdf")
        # Source file analysis
        if source_file is not None:
            source_data = SourceFile(Path(source_file))
            # Create empty arrays for the results
            g = np.zeros(shape=len(models), dtype=object)
            res = np.zeros(shape=len(models), dtype=object)
            # Caching the initial values of xmin and xmax
            xmin_init = kwargs.get("xmin", float("-inf"))
            xmax_init = kwargs.get("xmax", float("inf"))
            for i, model in enumerate(models):
                # Without a proper initialization of xmin and xmax the fit doesn't converge
                x_peak = source_data.hist.bin_centers()[source_data.hist.content.argmax()]
                if xmin_init == float("-inf"):
                    kwargs.update(xmin=x_peak - 0.5 * x_peak)
                if xmax_init == float("inf"):
                    kwargs.update(xmax=x_peak + 0.5 * x_peak)
                # Fit the spectrum in the given range and log
                fit_model = source_data.fit(model, **kwargs)
                if isinstance(fit_model, aptapy.models.Fe55Forest):
                    reference_energy: float = fit_model.energies[0]   # type: ignore [attr-defined]
                    line_adc = reference_energy / fit_model.status.correlated_pars[1]
                    sigma = fit_model.status.correlated_pars[2]
                elif isinstance(fit_model, aptapy.models.Gaussian):
                    line_adc = fit_model.status.correlated_pars[1]
                    sigma = fit_model.status.correlated_pars[2]
                else:
                    raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")
                g[i] = gain(w, capacity, line_adc, line_model.status.correlated_pars, e_peak)
                res[i] = energy_resolution(line_adc, sigma)
                logyaml.add_source_results(source_data.file_path.name, fit_model)
                logyaml.add_source_gain_res(source_data.file_path.name, g[i], res[i])
                # Source file plotting and saving
                if plot or save:
                    plt.figure(f"{source_data.file_path.name}")
                    plt.title(f"{int(source_data.voltage)} V {fit_model.name()}")
                    source_data.hist.plot()
                    label = f"{fit_model.name()}\nFWHM/E@{e_peak:.1f} keV: {res[i]} %"
                    fit_model.plot(label=label)
                    plt.xlim(fit_model.default_plotting_range())
                    plt.legend()
                    if save:
                        plt.savefig(logyaml.log_folder / source_data.file_path.name, format='pdf')
                    if not plot:
                        plt.close("all")
            return res, g
        return line_model.status.correlated_pars
    finally:
        if save:
            logyaml.save()


def analyze_folder(folder_name: str, models: Sequence[type[AbstractFitModel]], w: float,
                   capacity: float, e_peak: float, plot: bool = False, save: bool = False,
                   **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze a folder containing calibration pulse files and source data (spectrum) files. If
    multiple calibration files are present, the first in alphabetical order is taken. For each
    spectrum a fit of the emission line(s) is done using the model(s) specified. If multiple models
    are given, the fit is done using both of them, and for each result the energy resolution and
    the gain is calculated. 

    Arguments
    ----------
    folder_name : str
        Name of the folder to analyze. Please note that only the path after the data folder path
        must be given. The path is automatically added during the analysis.
    models : Tuple[AbstractFitModel]
        Model(s) to fit the source emission line.
    W : float
        W-value of the detector gas.
    capacity : float
        Capacity of the capacitance of the readout circuit of the detector.
    e_peak : float
        Energy of the main emission line of the source (in keV).
    plot : bool, optional
        If True, plot the figures of the analysis, by default False.
    save : bool, optional
        If True, save a log file and plots of the analysis, by default False.
    **kwargs : dict
        Refer to `fileio.SourceFile.fit`.

    Returns
    -------
    voltage, res, g : ArrayLike
        Returns arrays with the voltage, the energy resolution and the gain of each spectrum file.
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    # Start the logging
    logyaml = LogYaml()
    if save:
        logyaml.start_logging()
    # Open the folder and select the first calibration file
    data_folder = DataFolder(ANALYSIS_DATA / folder_name)
    pulse_data = data_folder.pulse_data[0]
    # Analyzing, plotting and saving the calibration file
    line_model, pulse_fig, line_fig = pulse_data.analyze_pulses()
    logyaml.add_pulse_results(pulse_data.file_path.name, line_model)
    if not plot:
        plt.close(pulse_fig)
        plt.close(line_fig)
    if save:
        pulse_fig.savefig(logyaml.log_folder / "cal_pulses.pdf", format="pdf")
        line_fig.savefig(logyaml.log_folder / "cal_fit.pdf", format="pdf")
    # Source files analysis
    source_data = data_folder.source_data
    voltage = np.array([file.voltage for file in source_data])
    # Create empty arrays for the results
    g = np.zeros(shape=len(models), dtype=object)
    res = np.zeros(shape=len(models), dtype=object)
    # Caching the initial values of xmin and xmax
    xmin_init = kwargs.get("xmin", float("-inf"))
    xmax_init = kwargs.get("xmax", float("inf"))
    # Iterating on the given models for the spectrum fit
    for i, model in enumerate(models):
        fit_models = []
        for source in source_data:
            # Without a proper initialization of xmin and xmax the fit doesn't converge
            x_peak = source.hist.bin_centers()[source.hist.content.argmax()]
            if xmin_init == float("-inf"):
                kwargs.update(xmin=x_peak - 0.5 * x_peak)
            if xmax_init == float("inf"):
                kwargs.update(xmax=x_peak + 0.5 * x_peak)
            # Fit the spectrum in the given range
            fit_models.append(source.fit(model, **kwargs))
            logyaml.add_source_results(source.file_path.name, fit_models[-1])

        # Collect fit parameters and fit models from the results
        correlated_pars = np.array([fit_model.status.correlated_pars for fit_model in fit_models])
        # Order and number of parameters differ based on the model
        if isinstance(fit_models[0], aptapy.models.Fe55Forest):
            reference_energy: float = fit_models[0].energies[0]   # type: ignore [attr-defined]
            line_adc = reference_energy / correlated_pars[:, 1]
            sigma = correlated_pars[:, 2]
        elif isinstance(fit_models[0], aptapy.models.Gaussian):
            line_adc = correlated_pars[:, 1]
            sigma = correlated_pars[:, 2]
        else:
            raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")
        # Calculate gain and energy resolution and store them in the previously created arrays
        g[i] = gain(w, capacity, line_adc, line_model.status.correlated_pars, KALPHA)
        res[i] = energy_resolution(line_adc, sigma)
        # Source files plotting and saving
        if plot or save:
            for j, _s in enumerate(source_data):
                fig = plt.figure(f"{_s.file_path.name}")
                plt.title(f"{int(voltage[j])} V {fit_models[j].name()}")
                _s.hist.plot()
                label = f"{fit_models[j].name()}\nFWHM@{e_peak:.1f} keV: \
                    {fit_models[j].fwhm()} ADC" # type: ignore [attr-defined]
                fit_models[j].plot(label=label)
                plt.xlim(fit_models[j].default_plotting_range())
                plt.legend()
                if save:
                    plt.savefig(logyaml.log_folder / _s.file_path.name, format='pdf')
                if not plot:
                    plt.close(fig)
    # Plot gain and energy resolution for each if the different models given
    gain_fig = plt.figure("Gain")
    for i, model in enumerate(models):
        plt.errorbar(voltage, unumpy.nominal_values(g[i]), unumpy.std_devs(g[i]), fmt="o",
                         label=f"{model.__name__}")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

    res_fig = plt.figure("Energy resolution")
    for i, model in enumerate(models):
        plt.errorbar(voltage, unumpy.nominal_values(res[i]), unumpy.std_devs(res[i]), fmt="o",
                         label=f"{model.__name__}")
    plt.xlabel("Voltage [V]")
    plt.ylabel("FWHM / E")
    plt.legend()
    if not plot:
        plt.close(gain_fig)
        plt.close(res_fig)
    if save:
        gain_fig.savefig(logyaml.log_folder / "gain.pdf", format="pdf")
        res_fig.savefig(logyaml.log_folder / "energy_resolution.pdf", format="pdf")
        # Save .txt files with the results of gain and energy resolution for each model given
        for i, model in enumerate(models):
            output = np.array([voltage, unumpy.nominal_values(g[i]), unumpy.std_devs(g[i]), \
                           unumpy.nominal_values(res[i]), unumpy.std_devs(res[i])]).T
            header = "voltage [v], gain, s_gain, resolution, s_resolution"
            np.savetxt(logyaml.log_folder / \
                       f"results_{folder_name.split('/')[-1]}_{model.__name__}.txt",
                       output, delimiter=",", header=header)
        logyaml.save()
    return voltage, res, g


def compare_folders(folder_names: tuple[str], model: type[AbstractFitModel], w: float,
                    capacity: float, e_peak: float, plot: bool = False, save: bool = False,
                    **kwargs) -> None:
    """Analyze the files in different folders and compare them. In particular, the gain and the
    energy resolution are calculated and plotted. The gain and the energy resolution are obtained
    with the script `analyze_folder`, using the model given to fit the emission line(s) in the
    spectrum.

    Arguments
    ----------
    folder_names : Tuple[str]
        Name of the folders to compare.
    model : AbstractFitModel
        Model to fit the source emission line.
    W : float
        W-value of the detector gas.
    capacity : float
        Capacity of the capacitance of the readout circuit of the detector.
    e_peak : float
        Energy of the main emission line of the source (in keV).
    plot : bool, optional
        If True, plot the figures of the analysis, by default False.
    save : bool, optional
        If True, save a log file and plots of the analysis, by default False.
    **kwargs : dict
        Refer to `fileio.SourceFile.fit`.

    Returns
    -------
    voltage, res, g : ArrayLike
        Returns arrays with the voltage, the energy resolution and the gain of each spectrum file.
    """
    # pylint: disable=invalid-unary-operand-type
    # Start logging
    logyaml = LogYaml()
    if save:
        logyaml.start_logging()
    # Create empty arrays to store the results of the analysis of each folder
    voltage = np.zeros(shape=len(folder_names), dtype=object)
    res = np.zeros(shape=len(folder_names), dtype=object)
    g = np.zeros(shape=len(folder_names), dtype=object)
    # Analyze each folder and store the results
    for i, folder_name in enumerate(folder_names):
        voltage[i], res[i], g[i] = analyze_folder(folder_name, [model], w, capacity, e_peak,
                                                  plot, save=save, **kwargs)
    # Plot the gain
    gain_fig = plt.figure("Gain comparison")
    # Add exceptions on the data points, based on the folder analyzed
    # This will be moved in a specific method, maybe with info reported in a file
    for i, folder_name in enumerate(folder_names):
        if folder_name == "251118":
            voltage[i] = np.append(voltage[i], [300, 310, 320])
            g[i][0] = np.append(g[i][0], unumpy.uarray([33.821245688904206, 40.060014190288435, \
                                                        47.85847481701872],
                          [0.004015431655164531, 0.003950437657169777, 0.0038697164947474323]))
        if folder_name == "251127":
            g_350 = g[i][0][voltage[i] == 350.]
            g[i][0] = g[i][0][voltage[i] != 350.]
            voltage[i] = voltage[i][voltage[i] != 350.]
            voltage[i] = np.append(voltage[i], 350.)
            g[i][0] = np.append(g[i][0], np.min(g_350))

        plt.errorbar(voltage[i], unumpy.nominal_values(g[i][0]), unumpy.std_devs(g[i][0]), fmt="o",
                    label=load_label(folder_name))
        # Fit the gain with an exponential function
        exp_model = aptapy.models.Exponential()
        exp_model.fit(voltage[i], unumpy.nominal_values(g[i][0]), sigma=unumpy.std_devs(g[i][0]),
                      absolute_sigma=True)
        exp_model.plot(label=f"scale: {-exp_model.scale.ufloat()} V", color=last_line_color())
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

    if not plot:
        plt.close(gain_fig)
    if save:
        gain_fig.savefig(logyaml.log_folder / "gain_comparison.pdf", format="pdf")


def analyze_trend(folder_name: str, model: type[AbstractFitModel], w: float, capacity: float,
                  e_peak: float, plot: bool = False, save: bool = False,
                  **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analyze a folder containing calibration pulse files and source data (spectrum) files. If
    multiple calibration files are present, the first in alphabetical order is taken. For each
    spectrum a fit of the emission line(s) is done using the given model. The gain and the
    resolution are calculated with the fit results and their trend with time and drift voltage
    is plotted. 

    Arguments
    ----------
    folder_name : str
        Name of the folder to analyze. Please note that only the path after the data folder path
        must be given. The path is automatically added during the analysis.
    model : AbstractFitModel
        Model to fit the source emission line.
    W : float
        W-value of the detector gas.
    capacity : float
        Capacity of the capacitance of the readout circuit of the detector.
    e_peak : float
        Energy of the main emission line of the source (in keV).
    plot : bool, optional
        If True, plot the figures of the analysis, by default False.
    save : bool, optional
        If True, save a log file and plots of the analysis, by default False.
    **kwargs : dict
        Refer to `fileio.SourceFile.fit`.

    Returns
    -------
    res, g, time, drift_voltage : ArrayLike
        Returns arrays with the energy resolution, gain, time and drift voltage.
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    # Start logging
    logyaml = LogYaml()
    if save:
        logyaml.start_logging()
    # Open the folder and select the first calibration file, we need the fit parameters for the
    # analysis of the gain using the escape peak
    data_folder = DataFolder(ANALYSIS_DATA / folder_name)
    pulse_data = data_folder.pulse_data[0]
    # Analyzing, without plotting and saving, that is done in analyze_folder. Also no need to log
    _, pulse_fig, line_fig = pulse_data.analyze_pulses()
    plt.close(pulse_fig)
    plt.close(line_fig)
    # Analyze the folder and take gain and resolution
    _, res, g = analyze_folder(folder_name, [model], w, capacity, e_peak, plot, save, **kwargs)
    res = res[0]
    g = g[0]
    # Extracting real times and drift voltage
    source_files = data_folder.source_data
    real_times = np.array([_source.real_time for _source in source_files])
    drift_voltage = np.array([_source.drift_voltage for _source in source_files])
    # Cumulating time for consecutive data
    time = real_times.cumsum()
    # Plotting and saving
    out_name = str(folder_name).rsplit('/', maxsplit=1)[-1]
    if plot or save:
        fig = plt.figure("Gain vs time")
        plt.errorbar(time, unumpy.nominal_values(g), unumpy.std_devs(g), fmt=".",
                     label=r"K$\alpha$")
        plt.xlabel("Time [s]")
        plt.ylabel("Gain")
        plt.legend()
        if save:
            plt.savefig(logyaml.log_folder / f"gain_time_{out_name}.pdf", format="pdf")
        if not plot:
            plt.close(fig)
    if plot or save:
        fig = plt.figure("Resolution vs time")
        plt.errorbar(time, unumpy.nominal_values(res), unumpy.std_devs(res), fmt=".",
                     label=r"K$\alpha$")
        plt.xlabel("Time [s]")
        plt.ylabel("FWHM/E")
        plt.legend()
        if save:
            plt.savefig(logyaml.log_folder / f"resolution_time_{out_name}.pdf", format="pdf")
        if not plot:
            plt.close(fig)
    if plot or save:
        fig = plt.figure("Gain vs drift")
        plt.errorbar(drift_voltage, unumpy.nominal_values(g), unumpy.std_devs(g), fmt=".",
                     label=r"K$\alpha$")
        plt.xlabel("Drift voltage [v]")
        plt.ylabel("Gain")
        plt.legend()
        if save:
            plt.savefig(logyaml.log_folder / f"gain_drift_{out_name}.pdf", format="pdf")
        if not plot:
            plt.close(fig)
    if plot or save:
        fig = plt.figure("Resolution vs drift")
        plt.errorbar(drift_voltage, unumpy.nominal_values(res), unumpy.std_devs(res), fmt=".",
                     label=r"K$\alpha$")
        plt.xlabel("Drift voltage [v]")
        plt.ylabel("FWHM/E")
        plt.legend()
        if save:
            plt.savefig(logyaml.log_folder / f"resolution_drift_{out_name}.pdf", format="pdf")
        if not plot:
            plt.close(fig)
    return res, g, time, drift_voltage
