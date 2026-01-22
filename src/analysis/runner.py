from pathlib import Path

from aptapy.plotting import plt

from .app import load_class
from .config import AppConfig
from .fileio import PulsatorFile, SourceFile, Folder
from .tasks import (
    calibration,
    fit_peak,
    gain_single,
    resolution_single,
    resolution_escape,
    plot_spec
)


SINGLE_TASK_REGISTRY = {
    "gain": gain_single,
    "resolution": resolution_single,
    "resolution_escape": resolution_escape,
    "plot": plot_spec
}

FOLDER_TASK_REGISTRY = {
    "gain": ""
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
    context = dict(config=config)
    if spec_fit_config is not None:
        source = SourceFile(Path(source_file_path), cal_results["model"])
        context["source"] = source
        results = dict()
        # Execute all fitting subtasks defined in the configuration
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
        # We skip plot here because we want all the results to be in the context
        task_exceptions = ["calibration", "spectrum_fitting", "plot"]
        if task.task in task_exceptions or getattr(task, "skip", False):
            continue
        func = SINGLE_TASK_REGISTRY.get(task.task)
        if func:
            # Remove the name of the task from the keyword arguments
            kwargs = task.model_dump(exclude={"task"})
            context = func(context, **kwargs)
    plot_config = config.plot
    if plot_config is not None:
        func = SINGLE_TASK_REGISTRY.get(plot_config.task)
        if func:
            kwargs = plot_config.model_dump(exclude={"task"})
            func(context, **kwargs)

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


