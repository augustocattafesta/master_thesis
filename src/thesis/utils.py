from pathlib import Path

import numpy as np
from aptapy.hist import Histogram1d
from hexsample.tasks import reconstruct, simulate
from hexsample.fileio import peek_readout_type, digi_input_file_class, ReconInputFile
from hexsample.readout import HexagonalReadoutRectangular
from hexsample.caldb import CalDB
from .defaults import DEFAULT_LAYOUT, DEFAULT_NUM_COLS, DEFAULT_NUM_ROWS, DEFAULT_PITCH, DEFAULT_PADDING

def create_sim_readout(enc: str, gain: str, pedestal: str):
    readout = HexagonalReadoutRectangular(
        layout=DEFAULT_LAYOUT,
        num_cols=DEFAULT_NUM_COLS,
        num_rows=DEFAULT_NUM_ROWS,
        pitch=DEFAULT_PITCH,
        padding=DEFAULT_PADDING,
        enc=CalDB.open_enc(enc),
        gain=CalDB.open_gain(gain),
        pedestal=CalDB.open_pedestal(pedestal)
    )
    return readout

def digi_file(input_path: str):
    readout_mode = peek_readout_type(input_path)
    file_type = digi_input_file_class(readout_mode)
    return file_type(str(input_path))


def generate_dataset(output_path, num_events, sensor, source, readout, overwrite=False, **kwargs):
    path = Path(output_path)
    if not path.exists() or overwrite:
        print(f"File {path.stem} not found, generating dataset.")
        simulate(
            output_file_path=str(path),
            num_events=num_events,
            sensor=sensor,
            source=source,
            readout=readout,
            **kwargs
            )
    else:
        print(f"File {path.stem} found, skipping dataset generation.")

    return digi_file(str(path))


def reconstruct_dataset(input_path, recon_method, zero_sup_threshold=0, overwrite=False, **kwargs):
    input_path = Path(input_path)
    suffix = kwargs.get("suffix", f"recon_{recon_method}")
    kwargs.pop("suffix", None)
    recon_path = input_path.parent / f"{input_path.stem}_{suffix}.h5"
    
    if not recon_path.exists() or overwrite:
        print(f"File {recon_path.stem} not found, reconstructing dataset.")
        reconstruct(
            input_file_path=str(input_path),
            suffix=suffix,
            pos_recon_algorithm=recon_method,
            zero_sup_threshold=zero_sup_threshold,
            max_neighbors=6,
            **kwargs
        )
    else:
        print(f"File {recon_path.stem} found, skipping reconstruction.")
    
    return ReconInputFile(str(recon_path))


def cluster_size_hist(recon_file, density=True):
    edges = np.arange(0.5, 7.5, 1)
    hist = Histogram1d(edges, xlabel="Cluster size")
    cluster_size = recon_file.column("cluster_size")
    content = np.histogram(cluster_size, bins=edges, density=density)[0]
    hist.set_content(content)

    return hist