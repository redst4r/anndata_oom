from .dataframe import load_obs, load_var
from .matrix import h5csr_into_mem_rows
import h5py
import numpy as np
import anndata


def _subset_sparse_into_mem(x_on_disk: h5py.Group, rows, cols=None):
    """subset a sparse matrix
    - rows or done oom
    - cols are done in mem
    """
    # TODO: check hat the group is really a Matrix
    x_tmp = h5csr_into_mem_rows(rows, x_on_disk)
    if cols is not None:
        x_tmp = x_tmp[:, cols]
    return x_tmp


def _subset_dense_into_mem(x_on_disk: h5py.Dataset, rows, cols=None):
    # TODO: check hat the group is really a Matrix
    x_tmp = x_on_disk[rows]
    if cols is not None:
        x_tmp = x_tmp[:, cols]
    return x_tmp


def _subset_collection(collection, rows, cols=None):
    """collection could be something like .obsm, obsp (i.e. a dict of arrays)"""
    # subset any layers
    res = {}
    for name in collection:
        layer = collection[name]
        if isinstance(layer, h5py.Group):
            # its a sparse matrix
            res[name] = _subset_sparse_into_mem(layer, rows, cols)
        elif isinstance(layer, h5py.Dataset):
            # its a numpy array
            res[name] = _subset_dense_into_mem(layer, rows, cols)

        else:
            raise ValueError("unknown h5py object")

    return res


def read_h5ad_row_subset(fname: str, rows: list):
    """
    reads am h5ad from disk, but only oads a subset of the rows

    TODO: currently doesnt copy
    - uns
    """
    var = load_var(fname)
    obs = load_obs(fname)

    assert set(obs.index) & set(rows) == set(rows), "cant find some rows"
    new_obs = obs.loc[rows]
    ix = np.where(  # all row numbers that we want
        obs.index.isin(rows)
    )[0]
    with h5py.File(fname) as f:
        # Subset X
        X = _subset_sparse_into_mem(f["/X"], ix)

        # subset any layers
        layers = _subset_collection(f["/layers"], ix)
        obsm = _subset_collection(f["/obsm"], ix)
        # obsp
        # Note: these are square matrices of n_obs x n_obs
        # and we need to do the subset on both rows and cols
        # - the first subset (on rows) we do out of mem
        # - the second one we do in mem
        obsp = _subset_collection(f["/obsp"], ix, ix)

        # varm is n_features x m
        # we dont want to subset, but need to cary if over
        # -> just subset to all cols
        varm = _subset_collection(f["/varm"], np.arange(var.shape[0]))
        varp = _subset_collection(f["/varp"], np.arange(var.shape[0]))

        adata = anndata.AnnData(
            X=X,
            obs=new_obs,
            var=var,
            obsm=obsm,
            obsp=obsp,
            layers=layers,
            varm=varm,
            varp=varp,
        )
        return adata
