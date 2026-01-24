from pathlib import Path

from .config import AppConfig
from .fileio import Folder, PulsatorFile, SourceFile
from .tasks import (
    calibration,
    compare_gain,
    drift,
    fit_peak,
    gain_folder,
    gain_single,
    gain_trend,
    plot_spectrum,
    resolution_escape,
    resolution_folder,
    resolution_single,
)
from .utils import load_class


SINGLE_TASK_REGISTRY = {
    "gain": gain_single,
    "resolution": resolution_single,
    "resolution_escape": resolution_escape,
    "plot": plot_spectrum
}


FOLDER_TASK_REGISTRY = {
    "gain": gain_folder,
    "gain_trend": gain_trend,
    "resolution": resolution_folder,
    "drift": drift,
    "plot": plot_spectrum
}


FOLDERS_TASK_REGISTRY = {
    "compare_gain": compare_gain
}


def run(
        config_file_path: str | Path,
        *paths: str | Path
        ) -> dict:
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

    Returns
    -------
    context : dict
        Dictionary containing all the info and results of the analysis pipeline.
    """
    # Load configuration file
    config = AppConfig.from_yaml(config_file_path)
    context = dict(config=config, sources={}, fit={}, results={}, figures={})
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
    # Load source files with the calculated calibration model
    calibration_model = context["calibration"]["model"]
    sources = [SourceFile(Path(p), calibration_model) for p in source_file_paths]
    context["sources"] = {str(s.file_path.stem): s for s in sources}
    # Run all fitting subtasks defined in the configuration file for each source file
    spec_fit_config = config.spectrum_fitting
    if spec_fit_config is not None:
        for source in sources:
            context["tmp_source"] = source  # Think how to avoid this
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
        if len(sources) == 1:
            func = SINGLE_TASK_REGISTRY.get(task.task)
        else:
            func = FOLDER_TASK_REGISTRY.get(task.task)
        if func:
            # Remove the name of the task from the keyword arguments
            kwargs = task.model_dump(exclude={"task"})
            context = func(context, **kwargs)
    return context


def run_folders(
        config_file_path: str | Path,
        *folder_paths: str | Path
        ) -> dict:
    """Run the analysis pipeline defined in the configuration file on multiple data folders.

    Arguments
    ---------
    config_file_path : str | Path
        Path to the configuration file.
    *folder_paths : str | Path
        Paths to the data folders.
    """
    # Load configuration file
    config = AppConfig.from_yaml(config_file_path)
    context = dict(config=config, folders={}, results={})
    # Execute the analysis pipeline for each folder
    for folder_path in folder_paths:
        folder_context = run(
            config_file_path,
            folder_path
        )
        context["folders"][folder_path] = folder_context
    # After that all the folders have been analyzed, we can run the folder-level tasks
    pipeline = sorted(config.pipeline, key=lambda t: 1 if t.task == "plot" else 0)
    for task in pipeline:
        if task.task not in FOLDERS_TASK_REGISTRY:
            continue
        func = FOLDERS_TASK_REGISTRY.get(task.task)
        if func:
            # Remove the name of the task from the keyword arguments
            kwargs = task.model_dump(exclude={"task"})
            context = func(context, **kwargs)
    return context
