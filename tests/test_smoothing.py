from scipy import sparse
from anndata_oom.smoothing import oom_smooth
import numpy as np
from anndata import AnnData
import h5py


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
        [0, 0, 2],  # 5
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
        assert np.all(oom_smooth(h5fh, [0]).toarray() == np.array([1, 2, 0]))

        # here it should look at cells 4,5
        assert np.all(
            oom_smooth(h5fh, [3], add_self=False).toarray() == np.array([0, 0, 2])
        )
        np.testing.assert_allclose(
            oom_smooth(h5fh, [3], add_self=True).toarray(),
            np.array([[0, 0, (1 + 2 + 2) / 3]]),
        )

        # here it should look at cells 3,5
        np.testing.assert_allclose(
            oom_smooth(h5fh, [4], add_self=True).toarray(),
            np.array([[0, 0, (1 + 2 + 2) / 3]]),
        )

        np.testing.assert_allclose(
            oom_smooth(h5fh, [4], add_self=False).toarray(), np.array([[0, 0, 3 / 2]])
        )
