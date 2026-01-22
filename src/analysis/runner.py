from pathlib import Path

from aptapy.plotting import plt

from .app import load_class
from .config import AppConfig
from .fileio import Folder, PulsatorFile, SourceFile
from .tasks import (
    calibration,
    fit_peak,
    gain_single,
    plot_spec,
    resolution_escape,
    resolution_single,
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


def run(
        config_file_path: str | Path,
        *paths: str | Path
        ) -> None:
    """Run the analysis pipeline defined in the configuration file on the given data files or
    folder.
    
    Arguments
    ----------
    config_file_path : str | Path
        Path to the configuration file.
    *paths : str | Path
        Paths to the data files or folder. If only one path is given, it is assumed to be a folder
        containing source files and at least a pulse file. Otherwise, the last path is assumed to
        be the pulse file and all preceding ones are source files.
    """
    # Load configuration file
    config = AppConfig.from_yaml(config_file_path)
    context = dict(config=config, results={})
    # If only one path is given, we assume it is a folder containing source files and a pulse file.
    # Otherwise, the last path is the pulse file and all preceding ones are source files.
    if len(paths) == 1:
        data_folder = Folder(Path(paths[0]))
        source_file_paths = data_folder.source_files
        pulse_file_path = data_folder.pulse_file
    else:
        source_file_paths = paths[:-1]
        pulse_file_path = paths[-1]
    # Run calibration task on the pulse file
    cal_config = config.calibration
    if cal_config is not None:
        pulse = PulsatorFile(Path(pulse_file_path))
        context["pulse"] = pulse
        context = calibration(
            context=context,
            charge_conversion=cal_config.charge_conversion,
            plot=cal_config.plot
        )
    else:
        raise RuntimeError("No calibration task found in configuration.")
    # Run all fitting subtasks defined in the configuration file for each source file
    spec_fit_config = config.spectrum_fitting
    if spec_fit_config is not None:
        calibration_model = context["calibration"]["model"]
        for source_file_path in source_file_paths:
            tmp_source = SourceFile(Path(source_file_path), calibration_model)
            context["source"] = tmp_source
            # Execute all fitting subtasks defined in the configuration file
            for subtask in spec_fit_config.subtasks:
                fit_pars = subtask.fit_pars.model_dump()
                context = fit_peak(
                    context=context,
                    subtask=subtask.subtask,
                    model_class=load_class(subtask.model),
                    **fit_pars
                )
    # Now we run all the tasks defined in the configuration file. The pipeline is sorted
    # so that the plotting task is always executed at the end (to compute resolution or gain).
    pipeline = sorted(config.pipeline, key=lambda t: 1 if t.task == "plot" else 0)    
    for task in pipeline:
        # Skip tasks already executed and those marked to be skipped
        # We skip plot here because we want all the results to be in the context
        core_tasks = ["calibration", "spectrum_fitting"]
        if task.task in core_tasks or getattr(task, "skip", False):
            continue
        # Determine if we are running a single-file or folder task and select the appropriate
        # function
        if len(source_file_paths) == 1:
            func = SINGLE_TASK_REGISTRY.get(task.task)
        else:
            func = FOLDER_TASK_REGISTRY.get(task.task)
        if func:
            # Remove the name of the task from the keyword arguments
            kwargs = task.model_dump(exclude={"task"})
            context = func(context, **kwargs)
    plt.show()
