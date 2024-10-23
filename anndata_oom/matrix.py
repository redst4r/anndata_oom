from scipy import sparse
import h5py
import tqdm
import numpy as np


def h5sparse_to_csr(group: h5py.Group):
    """turns the h5ad CSR-group into a scipy.sparse.csr_matrix"""
    assert group.attrs["encoding-type"] == "csr_matrix"
    assert group.attrs["encoding-version"] == "0.1.0"

    indices = group["indices"][:]
    data = group["data"][:]
    indptr = group["indptr"][:]
    A = sparse.csr_matrix((data, indices, indptr), shape=group.attrs["shape"])
    return A


def h5_iter_csr(Xgroup):
    """iterate over the rows of the CSR matrix encoded in `Xgroup`

    yields row, colix, data
    """

    nrow, ncols = Xgroup.attrs["shape"]
    assert Xgroup.attrs["encoding-type"] == "csr_matrix"

    h5_indptr = Xgroup["indptr"]
    h5_indices = Xgroup["indices"]
    h5_data = Xgroup["data"]

    for r in range(nrow):
        # colix, data = get_row_from_h5ad(h5_indptr, h5_indices, h5_data, r)
        colix, data = get_row(r, h5_indptr, h5_indices, h5_data)
        yield r, colix, data


def h5csr_into_mem_rows(rows, h5dataset):
    """
    pull a csr from h5 into memory, but only selecting certain rows
    """
    assert h5dataset.attrs["encoding-type"] == "csr_matrix"
    original_cols = h5dataset.attrs["shape"][
        1
    ]  # number of columns in the original matrix
    indptr_h5 = h5dataset["indptr"]
    indices_h5 = h5dataset["indices"]
    data_h5 = h5dataset["data"]

    indptr, indices, data = row_index_csr(rows, indptr_h5, indices_h5, data_h5)

    return sparse.csr_matrix((data, indices, indptr), shape=[len(rows), original_cols])


def get_row(row, indptr_h5: h5py.Dataset, indices_h5: h5py.Dataset, data_h5: h5py.Dataset):
    """
    get a row from a csr matrix, returns colum indices and data
    """
    a, b = indptr_h5[row:row+2] # TODO is this really correct?! ACtually yes! see next two lines
    # a2 = indptr_h5[row]
    # b2 = indptr_h5[row+1]
    # assert a == a2
    # assert b == b2

    col_ix = indices_h5[a:b]  # the colum indices of row
    data = data_h5[a:b]
    return col_ix, data


def row_index_csr(rows, indptr_h5: h5py.Dataset, indices_h5: h5py.Dataset, data_h5: h5py.Dataset):
    """
    with a matrix in CSR format, extract the given rows and form a new matrix
    """
    indptr = [0]
    indices = []
    data = []
    for r in rows:
        col_ix, d = get_row(r, indptr_h5, indices_h5, data_h5)
        indices.extend(col_ix)
        data.extend(d)
        indptr.append(indptr[-1] + len(col_ix))
    return indptr, indices, data


# def row_col_extract_csr(rows, cols):
#     """
#     with a matrix in CRS format, extract the rows, cols  into a new matrix
#     """
#     raise ValueError('not worling')
#     indptr = [0]
#     indices = []
#     data = []
#     cols_set = set(cols)
#     for r in tqdm.tqdm(rows):
#         col_ix, d = get_row(row=r)
#
#         # filter for only the columns we're itnerested
#         cdd = [(c,dd) for c, dd in zip(col_ix, d) if c in cols_set]
#         if cdd:  # can be empty!
#             c,dd = zip(*cdd)
#
#             indices.extend(c)
#             data.extend(dd)
#             indptr.append(indptr[-1]+len(c))
#
#     # this matrix will have a lot of empty columns
#     # reshape it to contain only the desired cols
#     c = csr_matrix((data, indices, indptr), shape=[len(rows), np.max(cols)])
#
#     c = c[:, cols]
#     return c


def create_empy_matrix(store: h5py.File, group_name: str, ncols: int):
    """create an empty CSR matrix in the h5 file, with the given number of columns
    :param store: h5py storage handle
    """

    group = store.create_group(group_name)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["shape"] = (0, ncols)
    group.attrs["encoding-version"] = "0.1.0"

    group.create_dataset("indices", shape=0, maxshape=(None,), chunks=True, dtype=int)
    group.create_dataset("indptr", shape=0, maxshape=(None,), chunks=True, dtype=int)
    group.create_dataset("data", shape=0, maxshape=(None,), chunks=True)
    return group


def _add_row_to_group(group: h5py.Group, indices, data):
    """adds a single row to a CSR matrix (in h5ad format)"""
    assert len(indices) == len(data)
    indptr = group["indptr"]
    if indptr.size == 0:
        indptr.resize(indptr.size + 1, axis=0)
        indptr[0] = 0

    last_ptr = indptr[-1]
    indptr.resize(indptr.size + 1, axis=0)
    indptr[-1] = last_ptr + len(indices)

    datah5 = group["data"]
    old_shape = datah5.shape[0]
    datah5.resize(old_shape + len(data), axis=0)
    datah5[old_shape:] = data

    indicesh5 = group["indices"]
    old_shape = indicesh5.shape[0]
    indicesh5.resize(old_shape + len(indices), axis=0)
    indicesh5[old_shape:] = indices

    oldshape = group.attrs["shape"]
    group.attrs["shape"] = (oldshape[0] + 1, oldshape[1])


def csr_transform_rows_oom(Xgroup, group_transform, row_transformer):
    """apply `row_transformer` to each row of the matrix
    :param row_transformer: a function which takes row_int, col_ix, data and spits out a new col_ix, data
    """

    assert group_transform.attrs["encoding-type"] == "csr_matrix"
    assert group_transform.attrs["encoding-version"] == "0.1.0"

    for row, colix, data in tqdm.tqdm(
        h5_iter_csr(Xgroup), total=Xgroup.attrs["shape"][0]
    ):
        transformed_colix, transformed_data = row_transformer(row, colix, data)
        _add_row_to_group(group_transform, transformed_colix, transformed_data)


def csr_matrix_subset_columns(Xgroup, subset_ix, target: h5py.Group, name: str):
    """perform as column subset on the h5-CSR, i.e. in memory it would correspond to
    `X[:, subset_ix]`

    we're using the row_transform API here
    """
    
    def _select_cols_transformer(row_ix: int, col_ix: np.ndarray, data: np.ndarray, subset_cols: dict):
        new_col = []
        new_data = []
        for c, d in zip(col_ix, data):
            if c in subset_cols:
                new_col_ix = subset_cols[c]
                new_col.append(new_col_ix)
                new_data.append(d)
        return new_col, new_data

    group_transform = create_empy_matrix(target, name, ncols=len(subset_ix))

    # maps from old (subsetted) columns to new 0,...,n
    subset_dict = {c: n for n, c in enumerate(subset_ix)}

    csr_transform_rows_oom(
        Xgroup,
        group_transform,
        lambda r, c, d: _select_cols_transformer(r, c, d, subset_dict),
    )
    return group_transform



def subset_variables_h5ad(Xgroup, vargroup, sub_ix, store, target_x_name, target_var_name):
    # subset the matrix, save to store[target_x_name]
    _gt = csr_matrix_subset_columns(Xgroup, sub_ix, target=store, name=target_x_name)

    # subset the var dataframe
    sub_vargroup = store.create_group(target_var_name)
    for colname in vargroup:
        new_entries = vargroup[colname][sub_ix]
        sub_vargroup.create_dataset(name=colname, data=new_entries)
