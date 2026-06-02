"""
dealign with the Anndata encoding of pandas.DataFrames
"""

import h5py
import numpy as np

from anndata._io.h5ad import read_dataframe


def load_dataframe(h5ad_filename: str, path: str):
    """
    loads any dataframe in the h5ad. path specifics the location within the
    h5ad (e.g. /raw/var, /obs, /var)
    """
    with h5py.File(h5ad_filename, "r") as f:
        df = read_dataframe(f[path])
    return df


def load_obs(h5ad_filename):
    """load .obs into memory"""
    return load_dataframe(h5ad_filename, "/obs")


def load_var(h5ad_filename):
    """load .var into memory"""
    return load_dataframe(h5ad_filename, "/var")


def add_column(df_group, colname, data, encoding_type: str):
    """adding a single column to the dataframe
    same as `df[colname] = data`
    """

    assert encoding_type in ["array", "string-array"], (
        f"unknown encoding_type {encoding_type}"
    )

    # first check the the data has the right shape
    for k in df_group.keys():
        assert df_group[k].shape == data.shape, (
            "trying to add a column with the wrong shape!"
        )

    col = df_group.create_dataset(colname, data=data)
    col.attrs["encoding-type"] = encoding_type  # array, string-array
    col.attrs["encoding-version"] = "0.2.0"

    order = df_group.attrs["column-order"]
    if (
        len(order) == 0
    ):  # otherwise we end up with some werid type U32 which h5 cant encode
        order = np.array([], dtype="object")
    order = np.append(order, [colname])
    df_group.attrs["column-order"] = order
    return col


def subset_rows_of_dataframe(
    sub_ix: list, df_group: h5py.Group, target_group: h5py.Group
):
    """do a subset of the rows of a dataframe, equivalent to pd.DataFrame().iloc[[x,y,z]]"""

    # copy the attrs of the dataframe
    for k, v in dict(df_group.attrs).items():
        target_group.attrs[k] = v

    for colname in df_group:
        # print("colname", colname)
        new_entries = df_group[colname][sub_ix]
        # print(new_entries)
        target_group.create_dataset(name=colname, data=new_entries)
        # copy all attribues over
        for k, v in dict(df_group[colname].attrs).items():
            target_group[colname].attrs[k] = v
