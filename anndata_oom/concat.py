import h5py
import numpy as np
import pandas as pd
from anndata._io.h5ad import read_dataframe
from sctools.misc import load_var
import anndata._io as io

"""
a bunch of code to merge h5ads on disk without loading things to memory

NOTE/WARNNG: this modifies one of the files !!INPLACE!!!
"""


def extend_1D_h5dataset(dataset: h5py.Dataset, to_extend: np.ndarray):
    """appends the array `to_extend` to the dataset (on disk)

    note: modifies the dataset inplace!!
    """
    assert len(dataset.shape) == 1, "only works for 1D datasets"
    assert len(to_extend.shape) == 1, "only works for 1D datasets"
    assert dataset.dtype == to_extend.dtype
    current_size = dataset.shape[0]
    added_size = to_extend.shape[0]

    # print(f'Extending {current_size} with {added_size} elements')
    # print(dataset.shape, to_extend.shape)
    # first, need to increase size of old data
    dataset.resize(current_size + added_size, axis=0)

    # copy the data
    dataset[current_size:] = to_extend
    # print(f'new shape {dataset.shape}')


def stack_h5_crs_ondisk(group_to_extend, new_group):
    """stacks two CSR matrices within hdf5, INPLACE
    /data
    /indices
    /intptr

    3 steps:
    1. simply concat the /data's
    2. simply concat the /indices's
    3. merge the intptr: basically needs to offset everything by the #rows of the "top" matrix
    """

    assert (
        group_to_extend.attrs["encoding-type"] == "csr_matrix"
    ), "1st group is NOT a CRS matrix"
    assert (
        new_group.attrs["encoding-type"] == "csr_matrix"
    ), "1st group is NOT a CRS matrix"

    # 1.
    extend_1D_h5dataset(group_to_extend.get("data"), new_group.get("data")[:])

    # 2.
    extend_1D_h5dataset(group_to_extend.get("indices"), new_group.get("indices")[:])

    # 3.
    # Joingin the pointers is a bit more tricky, each has size `#rows`+1.
    # - `m+1`, `n+1`  -> `n+m+1`
    # -
    indptr1 = group_to_extend.get("indptr")
    indptr2 = new_group.get("indptr")

    nrows1 = indptr1.shape[0] - 1
    nrows2 = indptr2.shape[0] - 1

    offset = indptr1[-1]  # imjportant: do BEFORE resizing
    new_ptr = indptr2[:] + offset

    indptr1.resize(nrows1 + nrows2 + 1, axis=0)
    assert indptr1[nrows1:].shape == new_ptr.shape
    indptr1[nrows1:] = new_ptr

    # there's an idnependent attr to keeps track of the shape of the array
    # which we need to update
    oldshape = group_to_extend.attrs["shape"]
    # print("old shape" , oldshape)
    group_to_extend.attrs["shape"] = (
        nrows1 + nrows2,
        oldshape[1],
    )  # NOTE: need to assign, inplace modification doesnt work (e.g.  group1.attrs['shape'][0]=1)
    # print("new shape" , group_to_extend.attrs['shape'])


def stack_adata_on_disk(f1, f2):
    """
    stacks the adatas on disk, MODIFIYNG the first
    """
    with h5py.File(f1, "r+") as store1, h5py.File(f2) as store2:
        # before we do anything, make sure the columns are compatible
        var1 = load_var(f1)
        var2 = load_var(f2)
        assert np.all(
            var1.index == var2.index
        ), "cant stack matrices, they have different columns"

        # guard against other things we dont want to deal with here
        assert len(store1["obsm"].keys()) == 0 & len(store2["obsm"].keys()) == 0
        assert len(store1["varm"].keys()) == 0 & len(store2["varm"].keys()) == 0
        assert len(store1["varp"].keys()) == 0 & len(store2["varp"].keys()) == 0
        assert len(store1["obsp"].keys()) == 0 & len(store2["obsp"].keys()) == 0
        assert len(store1["uns"].keys()) == 0 & len(store2["uns"].keys()) == 0

        ##################
        ## actual merging
        ##################

        # merge /X
        print("stacking /X")
        stack_h5_crs_ondisk(store1["/X"], store2["/X"])

        # merge /obs
        print("stacking /obs")
        _merge_obs(store1, store2)  # curently disabled for testing

        # var stays unchanged

        # merge other layers
        # TODO: check if they exist, otherwise cryptic error msg
        print("stacking /layers")
        # stack_h5_crs_ondisk(store1['/layers/ambiguous'], store2['/layers/ambiguous'])
        # stack_h5_crs_ondisk(store1['/layers/spliced'], store2['/layers/spliced'])
        # stack_h5_crs_ondisk(store1['/layers/unspliced'], store2['/layers/unspliced'])
        # print('stacking /raw')
        # stack_h5_crs_ondisk(store1['/raw/X'], store2['/raw/X'])

        # TODO guard against nonempty obsm, varm
        assert len(store1["obsm"].keys()) == 0 & len(store2["obsm"].keys()) == 0
        assert len(store1["varm"].keys()) == 0 & len(store2["varm"].keys()) == 0


def _merge_obs(f1, f2):
    """
    merge the .obs of the h5ad objects
    """
    obs1 = read_dataframe(f1["/obs"])
    obs2 = read_dataframe(f2["/obs"])

    obs = pd.concat([obs1, obs2])

    io.specs.write_elem(f1, "obs", obs)
