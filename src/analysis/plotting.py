from typing import Any

import numpy as np
from aptapy.modeling import AbstractFitModel
from aptapy.plotting import plt

from .fileio import SourceFile


def write_legend(label: str | None, *axs: plt.Axes | None, loc: str = "best" ) -> None:
    # pylint: disable=protected-access
    valid_axs: list[plt.Axes] = [ax for ax in axs if ax is not None]
    if not valid_axs:
        valid_axs = [plt.gca()]
    headers: list[Any] = []
    labels: list[str] = []
    zorders: list[float] = []
    for ax in valid_axs:
        _header, _label = ax.get_legend_handles_labels()
        headers.extend(_header)
        labels.extend(_label)
        zorders.append(int(ax.get_zorder()))
        zorders.append(ax.get_zorder())
    if len(valid_axs) > 1:
        valid_axs[0].set_zorder(max(zorders) + 1)
        valid_axs[0].set_frame_on(False)
    if label is not None:
        leg = valid_axs[0].legend(headers, labels, loc=loc, title=label, title_fontsize=11)
        leg._legend_box.align = "bottom"    # type: ignore[attr-defined]


def get_xrange(source: SourceFile, models: list[AbstractFitModel]) -> list[float]:
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


def get_label(task_labels: list[str] | None, target_context: dict) -> str | None:
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
        key_label = f"{task}_label"
        if key_label in target_context:
            task_label = target_context[f"{task}_label"]
            label += f"{task_label}\n"
    # Remove the trailing newline character
    label = label[:-1]
    return label
