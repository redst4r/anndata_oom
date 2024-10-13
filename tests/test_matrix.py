from scipy import sparse
from anndata_oom.matrix import get_row, row_index_csr
from anndata_oom.oom import oom_smooth
import numpy as np
from anndata import AnnData
import h5py


def test_get_row():
    a = [[0, 2, 0], [1, 1, 0], [0, 0, 0]]

    s = sparse.csr_matrix(a)

    col_ix, data = get_row(0, s.indptr, s.indices, s.data)
    assert col_ix == [1] and data == [2]

    col_ix, data = get_row(1, s.indptr, s.indices, s.data)
    np.testing.assert_allclose(col_ix, [0, 1])
    np.testing.assert_allclose(data, [1, 1])

    col_ix, data = get_row(2, s.indptr, s.indices, s.data)
    assert len(col_ix) == 0 and len(data) == 0


def test_row_index_csr():
    """
    just ensure that we get the same indexing as s[rows,]
    especially, make sure empty rows are not dropped
    """
    a = [[0, 2, 0], [1, 1, 0], [0, 0, 0], [1, 2, 0]]

    s = sparse.csr_matrix(a)

    row_tests = [
        [0, 1],
        [0, 2],  # empty row
        [0, 0, 3, 3, 2],  # resorting etc
    ]
    for rows in row_tests:
        indptr, indices, data = row_index_csr(rows, s.indptr, s.indices, s.data)
        q = sparse.csr_matrix((data, indices, indptr), shape=[len(rows), s.shape[1]])
        assert np.all(s[rows].A == q.A)


def test_smoothing():
    """
    simple scenario: two cluster of cells with identical expression within the cluster
    """
    a = [
        [1, 2, 0],  # 0
        [1, 2, 0],  # 1
        [1, 2, 0],  # 2
        [0, 0, 1],  # 3
        [0, 0, 2],  # 4
        [0, 0, 2],
    ]  # 5

    neigbourshoods = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]

    adata = AnnData(sparse.csr_matrix(a))
    adata.obsp["connectivities"] = sparse.csr_matrix(neigbourshoods)
    fname = "/tmp/pytest_sdfgsdtrghkjnr.h5ad"
    adata.write_h5ad(fname)

    with h5py.File(fname) as h5fh:
        # smoothing must not change anything here
        assert np.all(oom_smooth(h5fh, [0]).A == np.array([1, 2, 0]))

        # here it should look at cells 4,5
        assert np.all(oom_smooth(h5fh, [3]).A == np.array([0, 0, 2]))

        # here it should look at cells 3,5
        assert np.all(oom_smooth(h5fh, [4]).A == np.array([0, 0, 1.5]))
