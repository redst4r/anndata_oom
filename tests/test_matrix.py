from scipy import sparse
from anndata_oom.matrix import get_row, row_index_csr
import numpy as np

def test_get_row():

    a = [[0, 2, 0],
         [1, 1, 0],
         [0, 0, 0]]

    s = sparse.csr_matrix(a)

    col_ix, data = get_row(0, s.indptr, s.indices, s.data)
    assert col_ix == [1] and data == [2]

    col_ix, data = get_row(1, s.indptr, s.indices, s.data)
    np.testing.assert_allclose(col_ix, [0, 1] )
    np.testing.assert_allclose(data, [1,1] )

    col_ix, data = get_row(2, s.indptr, s.indices, s.data)
    assert len(col_ix) == 0 and len(data) == 0


def test_row_index_csr():

    """
    just ensure that we get the same indexing as s[rows,]
    especially, make sure empty rows are not dropped
    """
    a = [[0, 2, 0],
         [1, 1, 0],
         [0, 0, 0],
         [1, 2, 0]]

    s = sparse.csr_matrix(a)

    row_tests = [
        [0, 1],
        [0, 2],  # empty row
        [0, 0, 3, 3, 2]  # resorting etc
    ]
    for rows in row_tests:
        indptr, indices, data = row_index_csr(rows, s.indptr, s.indices, s.data)
        q = sparse.csr_matrix((data, indices, indptr), shape=[len(rows), s.shape[1]] )
        assert np.all(s[rows].A == q.A)
