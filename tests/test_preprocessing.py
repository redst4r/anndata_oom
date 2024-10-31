from scipy import sparse
from anndata_oom.matrix import (
    get_row,
    row_index_csr,
    csr_transform_rows_oom,
    create_empy_matrix,
)
from anndata_oom.oom import oom_smooth, oom_mean_var

import numpy as np
import anndata
import h5py
from anndata_oom.preprocessing import oom_processing
import pandas as pd
import scanpy as sc


def test_prep():

    n_cells = 1000
    n_genes = 50
    top_n = 100

    if False:
        means = np.random.uniform(100,10000, size=n_genes)
        variances = np.random.uniform(5,50, size=n_genes)
        a = np.random.normal(means, variances, size=(n_cells, n_genes)).astype(np.float32)
        mask = np.random.rand(n_cells, n_genes) > 0.5
        a = sparse.csr_matrix(a * mask)

        genenames = [f"g{i}" for i in range(n_genes)]
        adata = anndata.AnnData(a, var=pd.DataFrame({'hgnc_symbol': genenames}).set_index('hgnc_symbol'))
    else: 
        # import scanpy as sc
        # adata = sc.datasets.pbmc3k()
        adata = anndata.read_h5ad('tests/pbmc3k.h5ad')
        adata.var.index.name = 'hgnc_symbol'
        # adata = adata[:, :500].copy()

    fname = "/tmp/pytest_fsagn.h5ad"
    adata.write_h5ad(fname)


    outfile = f"{fname}.out.h5ad"
    with h5py.File(fname, 'r') as source, h5py.File(outfile, 'w') as target:
        oom_processing(source, target, top_n=top_n)

    adata_observed = anndata.read_h5ad(outfile)
    obs_hvg = adata_observed.var_names


    adata_expected = anndata.read_h5ad(fname)
    sc.pp.filter_genes(adata_expected, min_cells=1)
    sc.pp.normalize_total(adata_expected, target_sum=10_000)
    sc.pp.highly_variable_genes(adata_expected, n_top_genes=top_n, flavor='cell_ranger')

    adata_expected = adata_expected[:, adata_expected.var['highly_variable']].copy()
    # adata_expected = adata_expected[:, obs_hvg].copy()
    sc.pp.normalize_total(adata_expected, target_sum=10_000)
    sc.pp.log1p(adata_expected)
    sc.pp.scale(adata_expected)

    assert adata_observed.var_names.to_list() == adata_expected.var_names.to_list()

    np.testing.assert_allclose(
        adata_observed.X.toarray(),
         adata_expected.X,
         rtol=0.0002
    )
