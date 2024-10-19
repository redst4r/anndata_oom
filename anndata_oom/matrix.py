from scipy import sparse

def h5_iter_csr(Xgroup):
    """iterate over the rows of the CSR matrix encoded in `Xgroup`

    yields row, colix, data
    """

    nrow, ncols = Xgroup.attrs['shape']
    assert Xgroup.attrs['encoding-type'] == "csr_matrix"

    h5_indptr = Xgroup['indptr']
    h5_indices = Xgroup['indices']
    h5_data = Xgroup['data']

    for r in range(nrow):
        # colix, data = get_row_from_h5ad(h5_indptr, h5_indices, h5_data, r)
        colix, data = get_row(r, h5_indptr, h5_indices, h5_data)
        yield r, colix, data


def h5csr_into_mem_rows(rows, h5dataset):
    """
    pull a csr from h5 into memory, but only selecting certain rows
    """
    assert h5dataset.attrs['encoding-type'] == 'csr_matrix'
    original_cols = h5dataset.attrs['shape'][1]  # number of columns in the original matrix
    indptr_h5 = h5dataset['indptr']
    indices_h5 = h5dataset['indices']
    data_h5 = h5dataset['data']

    indptr, indices, data = row_index_csr(rows, indptr_h5, indices_h5, data_h5)

    return sparse.csr_matrix((data, indices, indptr), shape=[len(rows), original_cols])


def get_row(row, indptr_h5, indices_h5, data_h5):
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


def row_index_csr(rows, indptr_h5, indices_h5, data_h5):
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
