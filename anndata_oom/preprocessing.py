from anndata_oom.matrix import csr_transform_rows_oom, create_empy_matrix, subset_variables_h5ad
from anndata_oom.oom import _oom_mean_var
from anndata_oom.dataframe import add_column
import numpy as np
import pandas as pd
import h5py
from statsmodels import robust
import warnings
from scanpy.preprocessing._highly_variable_genes import _get_mean_bins, _get_disp_stats, _Cutoffs, _nth_highest


def _fn_normalize_per_cell(row_ix: int, col_ix: np.ndarray, data: np.ndarray):
    alpha = 10_000  # Transcript per 10k
    _sum = np.sum(data)
    data_norm = alpha * data / _sum
    return col_ix, data_norm


def _fn_normalize_log_per_cell(row_ix: int, col_ix: np.ndarray, data: np.ndarray):
    alpha = 10_000  # Transcript per 10k
    _sum = np.sum(data)
    data_norm = alpha * data / _sum
    data_lognorm = np.log1p(data_norm)
    return col_ix, data_lognorm


def annotate_dispersion(df, top_n):
    """
    adds dispersion and normalized dispersion
     to each gene (with known mean and var)

     from `scanpy/src/scanpy/preprocessing/_deprecated/highly_variable_genes.py`
    """

    df["mean_bin"] = pd.cut(
        df["means"],
        np.r_[-np.inf, np.percentile(df["means"], np.arange(10, 105, 5)), np.inf],
    )
    disp_grouped = df.groupby("mean_bin", observed=False)["dispersions"]
    disp_median_bin = disp_grouped.median()
    # the next line raises the warning: "Mean of empty slice"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mad_bin = disp_grouped.apply(robust.mad)
        df["dispersions_norm"] = (
            df["dispersions"].values - disp_median_bin[df["mean_bin"].values].values
        ) / disp_mad_bin[df["mean_bin"].values].values

    cutoff = df["dispersions_norm"].sort_values().values[-(top_n + 1)]
    df["highly_variable"] = df["dispersions_norm"] > cutoff
    return df



# had to redo this one, the original code expects an AnnData, but onlu to access adata.n_vars
def _subset_genes(
    n_vars,
    *,
    mean,
    dispersion_norm,
    cutoff: _Cutoffs | int,
):
    """Get boolean mask of genes with normalized dispersion in bounds."""
    if isinstance(cutoff, _Cutoffs):
        dispersion_norm = np.nan_to_num(dispersion_norm)  # similar to Seurat
        return cutoff.in_bounds(mean, dispersion_norm)
    n_top_genes = cutoff
    del cutoff

    if n_top_genes > n_vars:
        # logg.info("`n_top_genes` > `n_var`, returning all genes.")
        n_top_genes = n_vars
    disp_cut_off = _nth_highest(dispersion_norm, n_top_genes)
    # logg.debug(
        # f"the {n_top_genes} top genes correspond to a "
        # f"normalized dispersion cutoff of {disp_cut_off}"
    # )
    return np.nan_to_num(dispersion_norm, nan=-np.inf) >= disp_cut_off


def oom_processing(source_h5: h5py.File, target_h5: h5py.File, n_top_genes: int):
    """

    does on-disk processing of the `source_h5` storing the result in target_h5

    Steps:
    1. normalize per cell
    2. HVG selection
    3. normlize again
    4. log1p
    5. scale
    """
    # copy var and obs over
    source_h5.copy('/var', target_h5,  expand_refs=True)  # expand refs due to categories (each cat-Series has an attrs['categories'] with a reference to /var/__categories/..)    
    source_h5.copy('/obs', target_h5, expand_refs=True)

    print("norm")
    # do rownorm on the old data, store result in target
    group_transform = create_empy_matrix(target_h5, '/X', source_h5['/X'].attrs['shape'][1])
    csr_transform_rows_oom(source_h5['/X'], group_transform, _fn_normalize_per_cell)

    # # HVGs, determined on the normalized data
    print("hvg: mean/var")
    mean, var, sv = _oom_mean_var(group_transform)

    n_vars = mean.shape[0]
    print(f"n_vars {n_vars}")

    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean

    # print(mean, type(mean), mean.dtype)

    add_column(target_h5['/var'], 'means', data=mean, encoding_type='array')
    # target_h5.create_dataset("/var/means", data=mean)
    # target_h5['/var/means'].attrs['encoding-type'] = 'array'
    # target_h5['/var/means'].attrs['encoding-version'] = '0.2.0'

    add_column(target_h5['/var'], 'vars', data=var, encoding_type='array')
    # target_h5.create_dataset("/var/vars", data=var)
    # target_h5['/var/vars'].attrs['encoding-type'] = 'array'
    # target_h5['/var/vars'].attrs['encoding-version'] = '0.2.0'

    add_column(target_h5['/var'], 'dispersion', data=dispersion, encoding_type='array')
    # target_h5.create_dataset("/var/dispersion", data=dispersion)
    # target_h5['/var/dispersion'].attrs['encoding-type'] = 'array'
    # target_h5['/var/dispersion'].attrs['encoding-version'] = '0.2.0'

    # order = target_h5['/var'].attrs['column-order']
    # order = np.append(order, ['means', 'vars', 'dispersion'])
    # target_h5['/var'].attrs['column-order'] = order

    if False:
        df = pd.DataFrame(
            {"means": mean, "vars": var, "dispersions": dispersion},
            index=source_h5["/var/hgnc_symbol"][:],
        )
        df_filtered = df.query("means>1e-12").copy()
        df_filtered = annotate_dispersion(df_filtered, top_n=n_top_genes)

    else:
        # piggy-backing on scanpys implementation of HVG
        min_disp: float = 0.5,
        max_disp: float = np.inf,
        min_mean: float = 0.0125,
        max_mean: float = 3,
        span: float = 0.3,
        n_bins: int = 20,
        flavor = 'cell_ranger'

        cutoff = _Cutoffs.validate(
            n_top_genes=n_top_genes,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
        )

        df = pd.DataFrame(dict(zip(["means", "dispersions"], (mean, dispersion))))
        df = df.query("means>1e-12")   # TODO: the scanpy code doesnt do that!

        df["mean_bin"] = _get_mean_bins(df["means"], flavor, n_bins)
        disp_stats = _get_disp_stats(df, flavor)

        # actually do the normalization
        df["dispersions_norm"] = (df["dispersions"] - disp_stats["avg"]) / disp_stats["dev"]
        df["highly_variable"] = _subset_genes(
            n_vars,
            mean=mean,
            dispersion_norm=df["dispersions_norm"].to_numpy(),
            cutoff=cutoff,
        )
        df_filtered = df

    # # restrict the matrices to HVG
    print("hvg filter")
    HVG = list(df_filtered.query("highly_variable").index)
    print(HVG)
    sub_ix = [
        ix for ix, gene in enumerate(df.index) if gene in HVG
    ]  # which columns to keep
    assert len(HVG) == len(sub_ix)
    subset_variables_h5ad(
        target_h5["/X"], target_h5["/var"], sub_ix, target_h5, "/Xsub1", "/varsub1"
    )

    # bit clunky, since we cant subset inplace
    del target_h5["/X"]
    target_h5.move("/Xsub1", "/X")
    del target_h5["/var"]
    target_h5.move("/varsub1", "/var")

    add_column(target_h5['/var'], 'dispersions_norm', data=df_filtered.query("highly_variable")['dispersions_norm'], encoding_type='array')




    actual_top_n = target_h5["/X"].attrs['shape'][1]
    print("actual_top_n", actual_top_n)


     #TODO: explicitly create layers if it doesnt exist, adding encoding metadata
     # {'encoding-type': 'dict', 'encoding-version': '0.1.0'}

    # renormalize and log
    print("renorm + log")
    group_transform = create_empy_matrix(target_h5, "/layers/norm_log", actual_top_n)
    csr_transform_rows_oom(target_h5["/X"], group_transform, _fn_normalize_log_per_cell)
    
    # do scaling from /layers/norm_log into /layers/norm_log_scale
    # need to recompute the mean/var
    print("scale")
    m, v, _ = _oom_mean_var(group_transform)

    def row_trans_scale(row_ix: int, col_ix: np.ndarray, data: np.ndarray):
        x = np.zeros(actual_top_n)
        x[col_ix] = data

        x = (x - m) / np.sqrt(v)
        new_col_ix = np.arange(actual_top_n)
        newdata = x
        return new_col_ix, newdata

    group_transform = create_empy_matrix(target_h5, "/layers/norm_log_scale", actual_top_n)
    csr_transform_rows_oom(
        target_h5["/layers/norm_log"], group_transform, row_trans_scale
    )

    # print("shape X", target_h5['/Xsub1'].attrs['shape'])
    print("shape var", dict(target_h5['/var'].attrs))
    print("shape X", dict(target_h5['/layers/norm_log_scale'].attrs))
    print("var X", target_h5['/var/hgnc_symbol'])

    del target_h5["/X"]
    target_h5.move("/layers/norm_log_scale", "/X")

    return df_filtered
