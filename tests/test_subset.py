import anndata
import numpy as np
import scanpy as sc
from scipy import sparse

from anndata_oom.utils import compare_adatas
from anndata_oom.subset import read_h5ad_row_subset


def test_subset():

    adata = anndata.AnnData(
        X=sparse.rand(100, 40).tocsr(),
        obsm={
            "X_umap": np.random.rand(100, 40),
            "X_sparse": sparse.rand(100, 40).tocsr(),
        },
        obsp={
            "a": np.random.rand(100, 100),
            "b": sparse.rand(100, 100).tocsr(),
        },
        varm={"PC": np.random.rand(40, 3)},
        varp={
            "stuff": np.random.rand(
                40,
                40,
            ),
            "stuff2": sparse.rand(40, 40).tocsr(),
        },
    )
    sc.pp.neighbors(adata)

    fname = "/tmp/t.h5ad"
    adata.write_h5ad(fname)

    rows = ["1", "2"]
    a1 = read_h5ad_row_subset("/tmp/t.h5ad", rows=rows)
    a2 = anndata.read_h5ad("/tmp/t.h5ad")[rows].copy()

    assert compare_adatas(a1, a2)

    assert compare_adatas(
        read_h5ad_row_subset("/tmp/t.h5ad", rows=rows),
        anndata.read_h5ad("/tmp/t.h5ad")[rows].copy(),
    )
