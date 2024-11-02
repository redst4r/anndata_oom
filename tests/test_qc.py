import pandas as pd
import anndata
from anndata_oom.qc import do_qc_oom
import numpy as np
from scipy import sparse
def test_smoothing():
    """
    simple scenario: two cluster of cells with identical expression within the cluster
    """
    a = np.array([
        [1, 2, 0],  # 0
        [1, 2, 0],  # 1
        [0, 2, 1],  # 2
        [0, 0, 1],  # 3
        [0, 0, 2],  # 4
        [0, 0, 1],
    ])  # 5
    a = sparse.csr_matrix(a)

    adata = anndata.AnnData(
        a, 
        var=pd.DataFrame({'hgnc_symbol': ["A", "MT-1", "MT-2"]}).set_index('hgnc_symbol'),
        obs=pd.DataFrame({'index': [f"C{i}" for i in range(a.shape[0])]}).set_index('index')
    )

    fname = '/tmp/test_oom_qc.h5ad'
    adata.write_h5ad(fname)
    do_qc_oom(fname, use_raw=False)

    adata = anndata.read_h5ad(fname)

    np.testing.assert_allclose(
        adata.obs.n_molecules,
        [3,3,3,1,2,1]
    )

    np.testing.assert_allclose(
        adata.obs.n_genes,
        [2,2,2, 1,1,1]
    )

    np.testing.assert_allclose(
        adata.obs.percent_mito,
        [2/3, 2/3, 1, 1,1,1 ]
    )

    np.testing.assert_allclose(
        adata.var.n_counts,
        [2, 6, 5]
    )

    np.testing.assert_allclose(
        adata.var.n_cells,
        [2, 3, 4]
    )