"""_summary_
"""

import xraydb

def weighted_energy(element: str, *lines: str) -> float:
    """Compute the intensity-weighted mean energy of specified X-ray lines of an element.

    The data are retrieved from the X-ray database at https://xraydb.seescience.org/.

    Arguments
    ----------
    element : str
        Atomic symbol of the element.

    lines : str
        Lines to consider in the weighted mean.
    
    Returns
    -------
    energy : float
        Intensity-weighted mean energy of the lines.
    """
    line_data = [xraydb.xray_line(element=element, line=line) for line in lines]
    total_intensity = sum(l.intensity for l in line_data)

    return sum(l.energy * l.intensity for l in line_data) / total_intensity
