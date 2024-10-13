"""
Out of memory tricks for AnnData
"""

import h5py
import numpy as np
from scipy import sparse
from anndata_oom.matrix import h5csr_into_mem_rows
import tqdm
import collections
from sctools.misc import load_dataframe
import anndata


def subset_cells_h5ad_file(h5file: str, subset_ix, use_raw=False):
    """
    create an AnnData with a subset of the cells in the h5ad.
    """

    if use_raw:
        x_path = "/raw/X"
        var_path = "/raw/var"
    else:
        x_path = "/X"
        var_path = "/var"

    with h5py.File(h5file, "r") as f:
        X = h5csr_into_mem_rows(subset_ix, f[x_path])

    var = load_dataframe(h5file, var_path)
    df_obs = load_dataframe(h5file, "/obs")
    obs = df_obs.iloc[subset_ix]

    adata = anndata.AnnData(X, obs=obs, var=var)

    return adata


def batch(some_iterable, batchsize):
    """
    splits the iterable into a couple of chunks of size n
    handy for iterating over batches
    :param some_iterable:  iterable to be chunked/batched
    :param batchsize: batchSize
    :return: gnerator over iterables
    """
    # TODO this does not guard against np.arrays as they are also iterable (over single elements)
    assert isinstance(some_iterable, collections.abc.Iterable)
    length = len(some_iterable)
    for ndx in range(0, length, batchsize):
        yield some_iterable[ndx : min(ndx + batchsize, length)]


def oom_smooth(h5handle: h5py.File, cells, BATCHSIZE=1000, add_self=True):
    """
    apply smoothing to the expression matrix, given the neighborhood graph
    """

    """
    iterate over reqeusted cells in batches (for efficiency)
    pull out their neighbors transcriptomes
    smooth with the neighbors and store the result
    """
    h5conn = h5handle["/obsp/connectivities"]
    h5expression = h5handle["/X"]

    _res = []
    nbatches = int(np.ceil(len(cells) / BATCHSIZE))
    for rowbatch in tqdm.tqdm(batch(cells, BATCHSIZE), total=nbatches):
        # pull the connetivity graph for the requested cells into mem. this is NOT a square matrix
        connectivities = h5csr_into_mem_rows(rowbatch, h5conn)  # batchsize x totalCells

        """
        to add the datapoint itself into the smoothing, one can just sneak it
        into the connections matrix, as a self edge.
        The cells themselves will also end up in the `neighbor_cells` and their
        transcriptome will be pulled
        """
        if add_self:
            # connectivities[i, self_ix] is the identify: connect the cell to itself
            # setting it in the connecivities is expensive. instead create a new matrix and add it
            # for i, self_ix in enumerate(rowbatch):
            #    connectivities[i, self_ix] = 1

            adder_data = np.ones(len(rowbatch))
            adder_rows = range(len(rowbatch))
            adder_cols = rowbatch
            adder = sparse.csr_matrix(
                (adder_data, (adder_rows, adder_cols)), shape=connectivities.shape
            )

            connectivities = connectivities + adder

        # they neighor cell (index) are jsut the aggregated column indices here
        # some occur multiple times (if two cells in rowbatch share a neighbor)
        neighbor_cells = sorted(list(set(connectivities.indices)))
        neighbor_transcriptome = h5csr_into_mem_rows(
            neighbor_cells, h5expression
        )  # nNeighbors x Genes

        # nieghbor transcriptime is shape N x genes (N the number of neighbors)
        # but connectivities is len(rowbatch) x all_cells
        # hence, order/subset connectivities
        # this will be the basis for our smoothing matrix
        connectivities_subspace = connectivities[:, neighbor_cells]

        # standardize
        row_scaler = 1 / connectivities_subspace.sum(axis=1).A.flatten()
        normA = sparse.diags(row_scaler) @ connectivities_subspace

        # smooth
        s = normA @ neighbor_transcriptome
        _res.append(s)

    return sparse.vstack(_res)
