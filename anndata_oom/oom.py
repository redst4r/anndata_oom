"""
Out of memory tricks for AnnData
"""
import h5py
# import numpy as np
from scipy import sparse
from anndata_oom.matrix import h5csr_into_mem_rows
import tqdm
import collections


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
    l = len(some_iterable)
    for ndx in range(0, l, batchsize):
        yield some_iterable[ndx:min(ndx + batchsize, l)]


def oom_smooth(h5handle: h5py.File, cells, BATCHSIZE=1000):
    """
    apply smoothing to the expression matrix, given the neighborhood graph
    """

    """
    iterate over reqeusted cells in batches (for efficiency)
    pull out their neighbors transcriptomes
    smooth with the neighbors and store the result
    """
    h5conn = h5handle['/obsp/connectivities']
    h5expression = h5handle['/X']

    _res = []
    for rowbatch in tqdm.tqdm(batch(cells, BATCHSIZE)):
        # pull the connetivity graph for the requested cells into mem. this is NOT a square matrix
        connectivities = h5csr_into_mem_rows(rowbatch, h5conn)  # batchsize x totalCells

        # they neighor cell (index) are jsut the aggregated column indices here
        # some occur multiple times (if two cells in rowbatch share a neighbor)
        neighbor_cells = sorted(list(set(connectivities.indices)))
        neighbor_transcriptome = h5csr_into_mem_rows(neighbor_cells, h5expression)   # nNeighbors x Genes

        # nieghbor transcriptime is shape N x genes (N the number of neighbors)
        # but connectivities is len(rowbatch) x all_cells
        # hence, order connectivities
        # this will be the basis for our smoothing matrix
        connectivities_subspace = connectivities[:, neighbor_cells]

        """
        TODO/problem: usually we would add the datapoint itself into the smoothing
        """

        # standardize
        row_scaler = 1 / connectivities_subspace.sum(axis=1).A.flatten()
        normA = sparse.diags(row_scaler) @ connectivities_subspace

        # smooth
        s = normA @ neighbor_transcriptome
        _res.append(s)

    return sparse.vstack(_res)
