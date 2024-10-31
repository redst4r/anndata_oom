from anndata_oom.matrix import csr_transform_rows_oom, create_empy_matrix, subset_variables_h5ad
from anndata_oom.oom import _oom_mean_var
import numpy as np
import pandas as pd
import h5py
from statsmodels import robust
import warnings

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


def oom_processing(source_h5: h5py.File, target_h5: h5py.File, top_n: int):
    """
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

    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean

    target_h5.create_dataset("/var/means", data=mean)
    target_h5.create_dataset("/var/vars", data=var)
    target_h5.create_dataset("/var/dispersion", data=dispersion)

    df = pd.DataFrame(
        {"means": mean, "vars": var, "dispersions": dispersion},
        index=source_h5["/var/hgnc_symbol"][:],
    )
    df_filtered = df.query("means>1e-12").copy()
    df_filtered = annotate_dispersion(df_filtered, top_n=top_n)

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


    actual_top_n = target_h5["/X"].attrs['shape'][1]
    print("actual_top_n", actual_top_n)

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
