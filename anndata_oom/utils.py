import numpy as np
from scipy import sparse


def _comp_array(x1, x2):
    """compare two arrays *either dense or sparse) for equality"""
    assert type(x1) == type(x2)

    if x1.shape != x2.shape:
        return False

    if isinstance(x1, np.ndarray):
        return bool(np.all(x1 == x2))
    elif isinstance(x1, sparse.spmatrix):
        return bool((x1 != x2).data.sum() == 0)
    else:
        raise ValueError()


def compare_adatas(a1, a2):
    """
    compare adatas by comparing their components (.var, .obs etc)

    TODO: only checks if .obs (.var) indices match, not if contents match
    """
    return all(
        [
            _comp_array(a1.X, a2.X),
            # np.all(a1.obs.index == a2.obs.index),
            a1.obs.equals(a2.obs),
            # np.all(a1.var.index == a2.var.index),
            a1.var.equals(a2.var),
        ]
        + [_comp_array(a1.obsm[m], a2.obsm[m]) for m in a1.obsm]
        + [_comp_array(a1.obsp[m], a2.obsp[m]) for m in a1.obsp]
        + [_comp_array(a1.layers[m], a2.layers[m]) for m in a1.layers]
        + [_comp_array(a1.varm[m], a2.varm[m]) for m in a1.varm]
        + [_comp_array(a1.varp[m], a2.varp[m]) for m in a1.varp]
    )
