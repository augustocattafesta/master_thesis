"""Module to analyze data
"""

from pathlib import Path

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel, line_forest
from aptapy.plotting import last_line_color, plt
from aptapy.typing_ import ArrayLike
from uncertainties import unumpy

from . import ANALYSIS_DATA
from .fileio import DataFolder, PulsatorFile, SourceFile
from .log import LogManager, logger
from .utils import AR_ESCAPE, KALPHA, KBETA, energy_resolution, gain


@line_forest(KALPHA - AR_ESCAPE, KBETA - AR_ESCAPE)
class ArEscape(aptapy.models.GaussianForestBase):
    pass

def analyze_file(pulse_file: str | Path, source_file: str | Path,
                 models: tuple[AbstractFitModel], w: float, capacity: float,
                 e_peak: float, plot: bool = False, save: bool = False,
                 **kwargs) -> ArrayLike | tuple[float, float]:
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
    log = LogManager()
    if save:
        logger.enable("analyze")
        log_folder = log.start_logging()
        log_main = log.log_main()
        log_fit = log.log_fit()
        log.log_args()
    else:
        logger.disable("analyze")
        log_main = logger
        log_fit = logger
    # Pulse file analysis and plotting
    pulse_data = PulsatorFile(Path(pulse_file))
    line_pars, pulse_fig, line_fig = pulse_data.analyze_pulses()
    if not plot:
        plt.close(pulse_fig)
        plt.close(line_fig)
    if save:
        pulse_fig.savefig(log_folder / "cal_pulses.pdf", format="pdf")
        line_fig.savefig(log_folder / "cal_fit.pdf", format="pdf")
    # Source file analysis
    if source_file is not None:
        log_main.info("SOURCE FILE(S) ANALYZED:")
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
            log_fit.info("FIT RESULTS:\n")
            pars, fit_model = source_data.fit(model, **kwargs)
            if issubclass(model, aptapy.models.Fe55Forest):
                line_adc = fit_model.energies[0] / pars[1]
                sigma = pars[2]
            elif issubclass(model, aptapy.models.Gaussian):
                line_adc = pars[1]
                sigma = pars[2]
            else:
                raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")
            g[i] = gain(w, capacity, line_adc, line_pars, e_peak)
            res[i] = energy_resolution(line_adc, sigma)
            # Results logging
            log_main.info("\nSOURCE FILE RESULTS:")
            log_main.info(f"{'model:':<12} {fit_model.name()}")
            log_main.info(f"{'gain:':<12} {g[i]}")
            log_main.info(f"{'resolution:':<12} {res[i]} %\n")
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
                    plt.savefig(log_folder / source_data.file_path.name, format='pdf')
                if not plot:
                    plt.close("all")
        return res, g
    return line_pars


def analyze_folder(folder_name: str, models: tuple[AbstractFitModel], w: float, capacity: float,
                   e_peak: float, plot: bool = False, save: bool = False,
                   **kwargs) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
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
    log = LogManager()
    if save:
        logger.enable("analyze")
        log_folder = log.start_logging()
        log_main = log.log_main()
        log_fit = log.log_fit()
        log.log_args()
    else:
        logger.disable("analyze")
        log_main = logger
        log_fit = logger
    # Open the folder and select the first calibration file
    data_folder = DataFolder(ANALYSIS_DATA / folder_name)
    pulse_data = data_folder.pulse_data[0]
    # Analyzing, plotting and saving the calibration file
    line_pars, pulse_fig, line_fig = pulse_data.analyze_pulses()
    if not plot:
        plt.close(pulse_fig)
        plt.close(line_fig)
    if save:
        pulse_fig.savefig(log_folder / "cal_pulses.pdf", format="pdf")
        line_fig.savefig(log_folder / "cal_fit.pdf", format="pdf")
    # Source files analysis
    log_main.info("SOURCE FILE(S) ANALYZED:")
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
        # Log fit results in another log file
        log_fit.info(f"FIT RESULTS FOLDER: {folder_name}\n")
        results = []
        for source in source_data:
            # Without a proper initialization of xmin and xmax the fit doesn't converge
            x_peak = source.hist.bin_centers()[source.hist.content.argmax()]
            if xmin_init == float("-inf"):
                kwargs.update(xmin=x_peak - 0.5 * x_peak)
            if xmax_init == float("inf"):
                kwargs.update(xmax=x_peak + 0.5 * x_peak)
            # Fit the spectrum in the given range
            results.append(source.fit(model, **kwargs))
        log_main.info("")
        # Collect fit parameters and fit models from the results
        pars, fit_models = zip(*results, strict=True)
        pars = np.stack(pars)
        fit_models = list(fit_models)
        # Order and number of parameters differ based on the model
        if issubclass(model, aptapy.models.Fe55Forest):
            line_adc = fit_models[0].energies[0] / pars[:, 1]
            sigma = pars[:, 2]
        elif issubclass(model, aptapy.models.Gaussian):
            line_adc = pars[:, 1]
            sigma = pars[:, 2]
        else:
            raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")
        # Calculate gain and energy resolution and store them in the previously created arrays
        g[i] = gain(w, capacity, line_adc, line_pars, KALPHA)
        res[i] = energy_resolution(line_adc, sigma)
        # Source files plotting and saving
        if plot or save:
            for j, _s in enumerate(source_data):
                fig = plt.figure(f"{_s.file_path.name}")
                plt.title(f"{int(voltage[j])} V {fit_models[j].name()}")
                _s.hist.plot()
                label = f"{fit_models[j].name()}\nFWHM@{e_peak:.1f} keV: {fit_models[j].fwhm()} ADC"
                fit_models[j].plot(label=label)
                plt.xlim(fit_models[j].default_plotting_range())
                plt.legend()
                if save:
                    plt.savefig(log_folder / _s.file_path.name, format='pdf')
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
        gain_fig.savefig(log_folder / "gain.pdf", format="pdf")
        res_fig.savefig(log_folder / "energy_resolution.pdf", format="pdf")
        # Save .txt files with the results of gain and energy resolution for each model given
        for i, model in enumerate(models):
            output = np.array([voltage, unumpy.nominal_values(g[i]), unumpy.std_devs(g[i]), \
                           unumpy.nominal_values(res[i]), unumpy.std_devs(res[i])]).T
            header = "voltage [v], gain, s_gain, resolution, s_resolution"
            np.savetxt(log_folder / f"results_{folder_name.split("/")[-1]}_{model.__name__}.txt",
                       output, delimiter=",", header=header)
    return voltage, res, g


def compare_folders(folder_names: tuple[str], model: AbstractFitModel, w: float,
                    capacity: float, e_peak: float, plot: bool = False, save: bool = False,
                    **kwargs) -> tuple[ArrayLike, ArrayLike]:
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
    # Start logging
    log = LogManager()
    if save:
        logger.enable("analyze")
        log_folder = log.start_logging()
        log.log_args()
    else:
        logger.disable("analyze")
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
    # Add custom labels based on the folder analyzed
    labels = {"251118":"W2b 86.6 top-right", "251127":"W8b 86.6 top-left high rate",
              "251201/1000": "Drift 1000 V", "251201/1300": "Drift 1300 V"}
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
                    label=labels.get(folder_name, f"{str(folder_name)}"))
        # Fit the gain with an exponential function
        model = aptapy.models.Exponential()
        model.fit(voltage[i], unumpy.nominal_values(g[i][0]), sigma=unumpy.std_devs(g[i][0]),
                  absolute_sigma=True)
        model.plot(label=f"scale: {-model.scale.ufloat()} V", color=last_line_color())
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

    if not plot:
        plt.close(gain_fig)
    if save:
        gain_fig.savefig(log_folder / "gain_comparison.pdf", format="pdf")

    # We need to re-add the removed points

    # plt.figure("Energy resolution")
    # for i, folder_name in enumerate(folder_names):
    #     if folder_name == "251118":
    #         voltage[i] = voltage[i][:-3]
    #     elif folder_name == "251127":
    #         voltage[i] = np.append(voltage[i], 350.)
    #     plt.errorbar(voltage[i], unumpy.nominal_values(res[i][0]), unumpy.std_devs(res[i][0]),
    #                  fmt="o", label=f"")
    # plt.xlabel("Voltage [V]")
    # plt.ylabel("FWHM / E")
    # plt.legend()


def analyze_trend(folder_name: str, model: AbstractFitModel, w: float, capacity: float,
                  e_peak, plot: bool = False, save: bool = False,
                  **kwargs) -> tuple[ArrayLike, ArrayLike]:
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
    g, res, time, drift_voltage : ArrayLike
        Returns arrays with the gain, the energy resolution, time and drift voltage.
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    # Start logging
    log = LogManager()
    if save:
        logger.enable("analyze")
        log_folder = log.start_logging()
        log.log_args()
    else:
        logger.disable("analyze")
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
    logger.info("SOURCE FILES ANALYZED FOR THE ESCAPE PEAK:")
    source_files = data_folder.source_data
    real_times = np.array([_source.real_time for _source in source_files])
    drift_voltage = np.array([_source.drift_voltage for _source in source_files])
    # Cumulating time for consecutive data
    time = real_times.cumsum()
    # Analyze the escape peak with a single gaussian to estimate the gain
    # results = [_source.fit(aptapy.models.Gaussian, xmin=25, xmax=53, num_sigma_left=1.5,
    #                       num_sigma_right=1.5) for _source in source_files]
    # pars, _ = zip(*results)
    # pars = np.stack(pars)
    # line_adc = pars[:, 1]
    # g_esc = gain(w, capacity, line_adc, line_pars, 2.9)
    # Plotting and saving
    out_name = str(folder_name).rsplit('/', maxsplit=1)[-1]
    if plot or save:
        fig = plt.figure("Gain vs time")
        plt.errorbar(time, unumpy.nominal_values(g), unumpy.std_devs(g), fmt=".",
                     label=r"K$\alpha$")
        # plt.errorbar(time, unumpy.nominal_values(g_esc), unumpy.std_devs(g_esc), fmt=".",
        #              label="Esc. Peak")
        plt.xlabel("Time [s]")
        plt.ylabel("Gain")
        plt.legend()
        if save:
            plt.savefig(log_folder / f"gain_time_{out_name}.pdf", format="pdf")
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
            plt.savefig(log_folder / f"resolution_time_{out_name}.pdf", format="pdf")
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
            plt.savefig(log_folder / f"gain_drift_{out_name}.pdf", format="pdf")
        if not plot:
            plt.close(fig)
    if plot or save:
        fig = plt.figure("Resolution vs drift")
        plt.errorbar(drift_voltage, unumpy.nominal_values(res), unumpy.std_devs(res), fmt=".",
                     label=r"K$\alpha$")
        plt.xlabel("Drift voltage [v]")
        plt.ylabel("Gain")
        plt.legend()
        if save:
            plt.savefig(log_folder / f"resolution_drift_{out_name}.pdf", format="pdf")
        if not plot:
            plt.close(fig)
    return res, g, time, drift_voltage
