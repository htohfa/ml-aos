"""Function to load the table of batoid simulations."""
from glob import glob
import pandas as pd

def load_data(path: str = "/content/drive/My Drive/rubin_aos_batoid_sims 2/data", chunks:int=100):
    """Load the table of batoid simulations.
    
    Parameters
    ----------
    path : str, default="data"
        Path to where the data is stored.
    chunks : int, default=100
        Number of chunks to load, where each chunk corresponds to a different
        atmosphere. Must be between 1 and 100, inclusive.

    Returns
    -------
    pd.DataFrame
        Table of simulated data.
    """
    # get the list of all files
    files = glob(f"{path}/obs*.pkl")

    # check that we found the files
    if len(files) == 0:
        raise RuntimeError(
            f"Did not find data files at path '{path}'."
        )

    # select the chunks
    files = files[:chunks]

    # load and return
    return pd.concat([pd.read_pickle(file) for file in files])
