"""
Out of memory tricks for AnnData
"""

import h5py
import numpy as np
from scipy import sparse
from anndata_oom.matrix import h5csr_into_mem_rows, h5_iter_csr
import tqdm


def oom_mean_var(h5_store: h5py.File, use_raw: bool):
    """calculate the mean and variance over rows (axis=0) for the given h5ad

    :param h5_store: open h5Handle of an h5ad file
    :param use_raw: if true use /raw/X else /X
    """
    Xgroup = h5_store["/X"] if not use_raw else h5_store["/raw/X"]
    return _oom_mean_var(Xgroup)


def _oom_mean_var(Xgroup: h5py.Group):
    """calculate the mean and variance over rows (axis=0) for the given CSR-group
    see `Welford's online algorithm` in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    assert Xgroup.attrs["encoding-type"] == "csr_matrix"
    nrow, ncols = Xgroup.attrs["shape"]

    count_vec = np.zeros(ncols)
    mean_vec = np.zeros(ncols)
    M_vec = np.zeros(ncols)

    for _r, colix, data in tqdm.tqdm(h5_iter_csr(Xgroup), total=nrow):
        "note: data only includes non-zeros, but the streaming algo needs to see ALL data"
        count_vec += 1
        data_full = np.zeros(ncols)
        data_full[colix] = data

        delta = data_full - mean_vec
        mean_vec += delta / count_vec

        delta2 = data_full - mean_vec
        M_vec += delta * delta2

    (mean_vec, variance, sample_variance) = (
        mean_vec,
        M_vec / count_vec,
        M_vec / (count_vec - 1),
    )
    return mean_vec, variance, sample_variance
