"""Module to analyze data
"""

from collections.abc import Sequence
from pathlib import Path

import aptapy.models
import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import last_line_color, plt
from uncertainties import unumpy

from . import ANALYSIS_DATA
from .fileio import DataFolder, PulsatorFile, SourceFile, load_label
from .log import LogYaml
from .utils import KALPHA, amptek_accumulate_time, energy_resolution, gain


def analyze_file(pulse_file: str | Path, source_file: str | Path,
                 models: Sequence[type[AbstractFitModel]], w: float,
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
        _, pulse_fig, line_fig = pulse_data.analyze_pulses(fit_charge=False)
        charge_conversion_model, _, _ = pulse_data.analyze_pulses(fit_charge=True)
        logyaml.add_pulse_results(pulse_data.file_path.name, charge_conversion_model)
        if not plot:
            plt.close(pulse_fig)
            plt.close(line_fig)
        if save:
            pulse_fig.savefig(logyaml.log_folder / "cal_pulses.pdf", format="pdf")
            line_fig.savefig(logyaml.log_folder / "cal_fit.pdf", format="pdf")
        # Source file analysis
        if source_file is not None:
            source_data = SourceFile(Path(source_file), charge_conversion_model)
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
                    # These lines were used to check the consistency of different methods to estimate
                    # the resolution
                    # if fit_escape:
                    #     kwargs.update(xmin=1., xmax=1.3, num_sigma_left=1.5, num_sigma_right=1.5)
                    #     escape_model = source_data.fit(aptapy.models.Gaussian, **kwargs)
                    #     resolution = 2.355*sigma * (KALPHA*1e3 - 2970) / (line_adc - escape_model.mu.value) / (KALPHA*1e3) * 100
                    #     print(resolution)
                elif isinstance(fit_model, aptapy.models.Gaussian):
                    line_adc = fit_model.status.correlated_pars[1]
                    sigma = fit_model.status.correlated_pars[2]
                else:
                    raise ValueError("Model not valid. Choose between Gaussian and Fe55Forest")
                g[i] = gain(w, line_adc, e_peak)
                res[i] = energy_resolution(line_adc, sigma)
                logyaml.add_source_results(source_data.file_path.name, fit_model)
                logyaml.add_source_gain_res(source_data.file_path.name, g[i], res[i])
                # Source file plotting and saving
                if plot or save:
                    plt.figure(f"{source_data.file_path.name}")
                    plt.title(f"{int(source_data.voltage)} V {fit_model.name()}")
                    source_data.hist.plot()
                    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
                    label = fr"$\Delta$E/E@{e_peak:.1f} keV: {res[i]} %" + f"\nFWHM@{e_peak:.1f} keV: {fwhm} fC"
                    fit_model.plot(label=label)
                    # escape_model.plot(fit_output=True)
                    xmin, xmax = fit_model.default_plotting_range()
                    plt.xlim(0.5 * xmin, xmax)
                    plt.legend()
                    if save:
                        plt.savefig(logyaml.log_folder / source_data.file_path.name, format='pdf')
                    if not plot:
                        plt.close("all")
            return res, g
        return charge_conversion_model.status.correlated_pars
    finally:
        if save:
            logyaml.save()


def analyze_folder(folder_name: str, models: Sequence[type[AbstractFitModel]], w: float,
                   e_peak: float, plot: bool = False, save: bool = False,
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
    line_model, pulse_fig, line_fig = pulse_data.analyze_pulses(fit_charge=False)
    logyaml.add_pulse_results(pulse_data.file_path.name, line_model)
    _, charge_fig, _ = pulse_data.analyze_pulses(fit_charge=True)
    if not plot:
        plt.close(pulse_fig)
        plt.close(line_fig)
        plt.close(charge_fig)
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
        g[i] = gain(w, line_adc, KALPHA)
        res[i] = energy_resolution(line_adc, sigma)
        # Source files plotting and saving
        if plot or save:
            for j, _s in enumerate(source_data):
                fig = plt.figure(f"{_s.file_path.name}")
                plt.title(f"{int(voltage[j])} V {fit_models[j].name()}")
                _s.hist.plot()
                label = f"FWHM@{e_peak:.1f} keV: {fit_models[j].fwhm()} fC" # type: ignore [attr-defined]
                fit_models[j].plot(label=label)
                plt.xlim(fit_models[j].default_plotting_range()[0] / 2,
                         fit_models[j].default_plotting_range()[1])
                plt.legend()
                if save:
                    plt.savefig(logyaml.log_folder / _s.file_path.name, format='pdf')
                if not plot:
                    plt.close(fig)
    # Plot gain and fit with an exponential
    gain_fig = plt.figure("Gain")
    for i in range(len(models)):
        gain_model = aptapy.models.Exponential()
        gain_model.fit(voltage, unumpy.nominal_values(g[i]), sigma=unumpy.std_devs(g[i]),
                       absolute_sigma=True)
        plt.errorbar(voltage, unumpy.nominal_values(g[i]), unumpy.std_devs(g[i]), fmt=".",
                         label=load_label(folder_name))
        gain_model.plot(label=f"scale: {-gain_model.scale.ufloat()} V")
    plt.yscale("log")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()
    # Plot energy resolution
    res_fig = plt.figure("Energy resolution")
    for i in range(len(models)):
        plt.errorbar(voltage, unumpy.nominal_values(res[i]), unumpy.std_devs(res[i]), fmt="o",
                         label=load_label(folder_name))
        min_idx = unumpy.nominal_values(res[i]).argmin()
        # Annotate the value of the minimum energy resolution
        plt.annotate(f"{unumpy.nominal_values(res[i])[min_idx]:.2f}",
             xy=(voltage[min_idx], unumpy.nominal_values(res[i])[min_idx]),
             xytext=(0, 30), textcoords='offset points', ha='center', va='top', fontsize=12)
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
                    e_peak: float, plot: bool = False, save: bool = False,
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
        voltage[i], res[i], g[i] = analyze_folder(folder_name, [model], w, e_peak, plot, save=save,
                                                   **kwargs)
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
    plt.yscale("log")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Gain")
    plt.legend()

    # Plot the energy resolution
    resolution_fig = plt.figure("Energy resolution comparison")
    for i, folder_name in enumerate(folder_names):
        if folder_name == "251118":
            voltage[i] = voltage[i][:-3]
        elif folder_name == "251127":
            voltage[i] = np.append(voltage[i], 350.)
        plt.errorbar(voltage[i], unumpy.nominal_values(res[i][0]), unumpy.std_devs(res[i][0]),
                     fmt="o", label=load_label(folder_name))
    plt.xlabel("Voltage [V]")
    plt.ylabel("FWHM / E")
    plt.legend()
        
    if not plot:
        plt.close(gain_fig)
        plt.close(resolution_fig)
    if save:
        gain_fig.savefig(logyaml.log_folder / "gain_comparison.pdf", format="pdf")
        resolution_fig.savefig(logyaml.log_folder / "resolution_comparison.pdf", format="pdf")


def analyze_trend(folder_name: str, model: type[AbstractFitModel], w: float,
                  e_peak, plot: bool = False, save: bool = False,
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
    _, res, g = analyze_folder(folder_name, [model], w, e_peak, False, save, **kwargs)
    res = res[0]
    g = g[0]
    # Extracting real times and drift voltage
    source_files = data_folder.source_data
    start_times = np.array([_source.start_time for _source in source_files])
    real_times = np.array([_source.real_time for _source in source_files])
    drift_voltage = np.array([_source.drift_voltage for _source in source_files])
    # Calculating the time 
    time_array = amptek_accumulate_time(start_times, real_times)

    # Analyze the escape peak with a single gaussian to estimate the gain
    # results = [_source.fit(aptapy.models.Gaussian, xmin=25, xmax=53, num_sigma_left=1.5,
    #                       num_sigma_right=1.5) for _source in source_files]
    # pars, _ = zip(*results)
    # pars = np.stack(pars)
    # line_adc = pars[:, 1]
    # g_esc = gain(w, line_adc, line_pars, 2.9)
    # Plotting and saving

    constant0 = aptapy.models.Constant()
    constant0.set_parameters(40.)
    stretched_exp = aptapy.models.StretchedExponential() + constant0
    stretched_exp.set_parameters(25, 2500/3600, 0.6)
    stretched_exp.fit(time_array, unumpy.nominal_values(g), sigma=unumpy.std_devs(g),
                      absolute_sigma=True, xmax=12000/3600)

    def stretched_exp_derivative(x, prefactor, scale, stretch):
        d_prefactor = - prefactor * stretch / scale**stretch
        # /60 to express the derivative in minutes
        return d_prefactor * x**(stretch - 1) * np.exp(-(x / scale)**stretch) / 60

    # constant1 = aptapy.models.Constant()
    # constant1.value.freeze(40.)
    # discharge_exp = aptapy.models.StretchedExponentialComplement(location=11500)
    # discharge_exp.set_parameters(25., 10000., 0.5)
    # discharge_exp = discharge_exp + constant1
    # discharge_exp.set_plotting_range(11500, 100000)
    # discharge_exp.fit(time_array, unumpy.nominal_values(g), sigma=unumpy.std_devs(g),
    #                   absolute_sigma=True, xmin=11776)

    out_name = str(folder_name).rsplit('/', maxsplit=1)[-1]
    if plot or save:
        fig = plt.figure("Gain vs time")
        plt.errorbar(time_array, unumpy.nominal_values(g), unumpy.std_devs(g), fmt=".",
                     label=load_label(folder_name))
        stretched_exp.plot(label=f"Charging:\nscale: {stretched_exp.status.correlated_pars[1]} s",
                           plot_components=False)
        # discharge_exp.plot(
        #     label=f"Discharging:\nscale: {discharge_exp.status.correlated_pars[1]} s",
        #     plot_components=False)
        # plt.errorbar(time, unumpy.nominal_values(g_esc), unumpy.std_devs(g_esc), fmt=".",
        #              label="Esc. Peak")
        plt.xlabel("Time [h]")
        plt.ylabel("Gain")
        plt.legend()

        derivative_fig = plt.figure("1/G dG/dt")
        tt = np.linspace(*stretched_exp.plotting_range(), 1000)
        yy = stretched_exp_derivative(tt, *stretched_exp.parameter_values()[:3]) / stretched_exp(tt)
        gg = stretched_exp(tt)
        gg /= max(gg)

        plt.plot(gg, yy, label=load_label(folder_name))
        plt.xlabel("Normalized gain")
        plt.ylabel("1/G dG/dt [1/min]")
        plt.tight_layout()
        plt.legend()

        if save:
            fig.savefig(logyaml.log_folder / f"gain_time_{out_name}.pdf", format="pdf")
            derivative_fig.savefig(logyaml.log_folder / f"derivative_gain_{out_name}.pdf",
                                   format="pdf")
        if not plot:
            plt.close(fig)
    if plot or save:
        fig = plt.figure("Resolution vs time")
        plt.errorbar(time_array, unumpy.nominal_values(res), unumpy.std_devs(res), fmt=".",
                     label=load_label(folder_name))
        plt.xlabel("Time [h]")
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
        plt.ylabel("Gain")
        plt.legend()
        if save:
            plt.savefig(logyaml.log_folder / f"resolution_drift_{out_name}.pdf", format="pdf")
        if not plot:
            plt.close(fig)
    return res, g, time_array, drift_voltage


from .config import AppConfig

def run_analysis(config_path: str):
    config = AppConfig.from_yaml(config_path)
    
