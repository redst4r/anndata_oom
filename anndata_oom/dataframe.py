"""
dealign with the Anndata encoding of pandas.DataFrames
"""
import h5py
import numpy as np


def add_column(df_group, colname, data, encoding_type:str):
    """adding a single column to the dataframe
    same as `df[colname] = data`
    """

    # first check the the data has the right shape
    for k in df_group.keys():
        assert df_group[k].shape == data.shape, "trying to add a column with the wrong shape!"

    col = df_group.create_dataset(colname, data=data)
    col.attrs['encoding-type'] = encoding_type  # array, string-array
    col.attrs['encoding-version'] = '0.2.0'

    order = df_group.attrs['column-order']
    if len(order) == 0:   # otherwise we end up with some werid type U32 which h5 cant encode
        order = np.array([],dtype='object')
    order = np.append(order, [colname])
    df_group.attrs['column-order'] = order
    return col


def subset_rows_of_dataframe(sub_ix: list, df_group: h5py.Group, target_group: h5py.Group):
    
    """ do a subset of the rows of a dataframe, equivalent to pd.DataFrame().iloc[[x,y,z]]
    """

    #copy the attrs of the dataframe
    for k,v in dict(df_group.attrs).items():
        target_group.attrs[k] = v

    for colname in df_group:
        # print("colname", colname)
        new_entries = df_group[colname][sub_ix]
        # print(new_entries)
        target_group.create_dataset(name=colname, data=new_entries)
        # copy all attribues over
        for k,v in dict(df_group[colname].attrs).items():
            target_group[colname].attrs[k] = v
