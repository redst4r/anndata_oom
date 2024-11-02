""" doing some of the QC metrics OOM
"""
# from toolz.itertoolz import sliding_window
import numpy as np
import h5py
from anndata._io.h5ad import read_dataframe
import itertools
from anndata_oom.dataframe import add_column
from anndata_oom.matrix import h5_iter_csr

def do_qc_oom(h5ad_file: str, use_raw:False):
    """estimate some QC metrics on-disk:
    - n_molecules: the number of counts per cells
    - n_genes: the number of genes per cell
    - percent_mito

    - gene_count: the number of counts per gene
    - gene_count_cell: the number of cells expressing the gene
    """
    with h5py.File(h5ad_file, 'r+') as store:
        matrix = store['/X'] if not use_raw else store['/raw/X']
        var = store['/var'] if not use_raw else store['/raw/var']
        obs = store['/obs']

        mito_gene_ix = set(
            np.where(
                read_dataframe(var).index.map(
                    lambda gene: gene.startswith('MT-')).values
            )[0])

        ncells, ngenes = store['/X'].attrs['shape']
        gene_counts = np.zeros(ngenes)
        n_molecules = np.zeros(ncells)
        n_mito = np.zeros(ncells)
        n_genes = np.zeros(ncells)
        gene_counts_ncells = np.zeros(ngenes)

        for cell_id, col_indices, data in h5_iter_csr(matrix):
            gene_counts_ncells[col_indices]+=1
            gene_counts[col_indices]+=data

            n_molecules[cell_id] = sum(data)
            n_genes[cell_id] = len(data)
            # only take the columns corresponding to mito-genes
            n_mito[cell_id] = sum(itertools.compress(data, [_ in mito_gene_ix for _ in col_indices]))

        add_column(obs, 'n_molecules', n_molecules, encoding_type='array')
        add_column(obs, 'n_genes', n_genes, encoding_type='array')
        add_column(obs, 'percent_mito', n_mito / n_molecules, encoding_type='array')

        add_column(var, 'n_counts', gene_counts, encoding_type='array')
        add_column(var, 'n_cells', gene_counts_ncells, encoding_type='array')
