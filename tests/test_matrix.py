from scipy import sparse
from anndata_oom.matrix import (
    get_row,
    row_index_csr,
    csr_transform_rows_oom,
    create_empy_matrix, subset_variables_h5ad
)
from anndata_oom.oom import oom_smooth, oom_mean_var
import anndata
import numpy as np
from anndata import AnnData
import h5py


def test_get_row():
    a = [[0, 2, 0], [1, 1, 0], [0, 0, 0]]

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
        assert np.all(s[rows].toarray() == q.toarray())


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
        assert np.all(oom_smooth(h5fh, [0]).toarray() == np.array([1, 2, 0]))

        # here it should look at cells 4,5
        assert np.all(oom_smooth(h5fh, [3]).toarray() == np.array([0, 0, 2]))

        # here it should look at cells 3,5
        assert np.all(oom_smooth(h5fh, [4]).A == np.array([0, 0, 1.5]))
        # assert np.all(oom_smooth(h5fh, [4]).toarray() == np.array([0, 0, 1.5]))


def get_random_matrix(nrows, ncols, density):
    a = sparse.random_array((nrows, ncols), density=density, format='csr') 
    a.data = a.data * 1000
    # a = a.astype(int)
    return a


def test_mean_var():
    a = get_random_matrix(nrows=1000, ncols=10, density=0.0001)
    a = sparse.csr_matrix(a)

    adata = AnnData(X=a)
    fname = "/tmp/pytest_sdfgsdtrghkjndfasdfag.h5ad"
    adata.write_h5ad(fname)

    with h5py.File(fname) as h5fh:
        m, v, sv = oom_mean_var(h5fh["/X"], use_raw=False)
        print(f"m: {m}")
        # smoothing must not change anything here
        np.testing.assert_allclose(m, a.toarray().mean(0))
        np.testing.assert_allclose(v, np.var(a.toarray(), 0))


def test_row_transform():
    a = np.array(
        [
            [1, 2, 0],  # 0
            [0, 4, 0],  # 1
            [2, 3, 0],
        ]
    )

    adata = AnnData(sparse.csr_matrix(a))
    fname = "/tmp/pytest_agsrqer3.h5ad"
    adata.write_h5ad(fname)

    with h5py.File(fname, mode="r+") as store:
        source = store["/X"]
        target = create_empy_matrix(store, "/layers/lognorm", source.attrs["shape"][1])

        def transformer(r, col_ix, data):  # just the identiyu
            return col_ix, data

        csr_transform_rows_oom(
            source,
            target,
            transformer,
        )

        assert np.all(source["indices"][:] == target["indices"][:])
        assert np.all(source["indptr"][:] == target["indptr"][:])
        assert np.all(source["data"][:] == target["data"][:])
        assert np.all(source.attrs["shape"] == target.attrs["shape"])


# assert np.all(group_normed['data'][:] == store['/X/data'][:])
import pandas as pd

def test_subset_variables_h5ad():
    n_genes = 10
    a = get_random_matrix(nrows=1000, ncols=n_genes, density=0.9)
    a = sparse.csr_matrix(a)

    adata = AnnData(X=a, var=pd.DataFrame({'hgnc_symbol': [f"g{i}" for i in range(n_genes)]}).set_index('hgnc_symbol'))
    fname = "/tmp/test_subset_variables.h5ad"
    adata.write_h5ad(fname)


    sub_ix = [0,2, 4]

    with h5py.File(fname, 'r+') as source:
        subset_variables_h5ad(
            source["/X"], source["/var"], sub_ix, source, "/Xsub1", "/varsub1"
        )

        # bit clunky, since we cant subset inplace
        del source["/X"]
        source.move("/Xsub1", "/X")
        del source["/var"]
        source.move("/varsub1", "/var")

    adata_observed = anndata.read_h5ad(fname)
    adata_expected = adata[:, sub_ix].copy()

    # print(adata_observed.var_names)
    # print(adata.var_names)
    assert np.all(adata_observed.var_names == adata_expected.var_names)
    # assert np.all((adata_observed.X == adata_expected.X).toarray())
    np.testing.assert_allclose(
        adata_observed.X.toarray(),
        adata_expected.X.toarray()
        )