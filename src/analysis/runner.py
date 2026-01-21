from pathlib import Path

from aptapy.plotting import plt

from .app import load_class
from .config import AppConfig
from .fileio import PulsatorFile, SourceFile, Folder
from .tasks import (
    calibration,
    fit_peak,
    gain_analysis,
)


TASK_REGISTRY = {
    "gain": gain_analysis
}


def run_single(source_file_path: str | Path,
               pulse_file_path: str | Path,
               config_file_path: str | Path
               ) -> None:
    # Load configuration file
    config = AppConfig.from_yaml(config_file_path)
    # Run calibration task
    cal_config = config.calibration
    if cal_config is not None:
        pulse = PulsatorFile(Path(pulse_file_path))
        cal_results = calibration(
            pulse=pulse,
            charge_conversion=cal_config.charge_conversion,
            plot=cal_config.plot
        )
    else:
        raise RuntimeError("No calibration task found in configuration.")
    # Run spectrum fitting tasks
    spec_fit_config = config.spectrum_fitting
    context = dict(config=config, line_vals=[], voltage=[])
    if spec_fit_config is not None:
        source = SourceFile(Path(source_file_path), cal_results["model"])
        # Execute all fitting subtasks defined in the configuration
        results = dict()
        for subtask in spec_fit_config.subtasks:
            fit_pars = subtask.fit_pars
            fit_results = fit_peak(
                source=source,
                model_class=load_class(subtask.model),
                xmin=fit_pars.xmin,
                xmax=fit_pars.xmax,
                num_sigma_left=fit_pars.num_sigma_left,
                num_sigma_right=fit_pars.num_sigma_right,
                absolute_sigma=fit_pars.absolute_sigma
            )
            results[subtask.subtask] = fit_results
    context["results"] = results
    # Now we run all the tasks defined in the configuration file
    for task in config.pipeline:
        # Skip tasks already executed and those marked to be skipped
        if task.task in ["calibration", "spectrum_fitting"] or getattr(task, "skip", False):
            continue
        func = TASK_REGISTRY.get(task.task)
        if func:
            # Remove the name of the task from the arguments
            kwargs = task.model_dump(exclude={"task"})
            _ = func(context, **kwargs)

    plt.show()


def run_folder(folder_path: str | Path,
               config_file_path: str | Path
               ) -> None:
    # Load configuration file
    config = AppConfig.from_yaml(config_file_path)
    # Load data folder
    data_folder = Folder(Path(folder_path))
    # Iterate over all source files in the folder
    cal_config = config.calibration
    if cal_config is not None:
        pulse = PulsatorFile(Path(data_folder.pulse_file))
        cal_results = calibration(
            pulse=pulse,
            charge_conversion=cal_config.charge_conversion,
            plot=cal_config.plot
        )
    else:
        raise RuntimeError("No calibration task found in configuration.")
    # Run spectrum fitting task for all source files
    spec_fit_config = config.spectrum_fitting
    if spec_fit_config is not None:
        for source_path in data_folder.source_files:
            source = SourceFile(Path(source_path), cal_results["model"])
            for subtask in spec_fit_config.subtasks:
                fit_pars = subtask.fit_pars
                fit_results = fit_peak(
                    source=source,
                    model_class=load_class(subtask.model),
                    xmin=fit_pars.xmin,
                    xmax=fit_pars.xmax,
                    num_sigma_left=fit_pars.num_sigma_left,
                    num_sigma_right=fit_pars.num_sigma_right,
                    absolute_sigma=fit_pars.absolute_sigma
                )


