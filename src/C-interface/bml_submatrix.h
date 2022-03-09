/** \file */

#ifndef __BML_SUBMATRIX_H
#define __BML_SUBMATRIX_H

#include "bml_types.h"

// Determine element indices for submatrix, given a set of nodes.
void bml_matrix2submatrix_index(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int *nodelist,
    int nsize,
    int *core_halo_index,
    int *vsize,
    int double_jump_flag);

// Determine core+halo indices from graph only
void bml_matrix2submatrix_index_graph(
    bml_matrix_t * B,
    int *nodelist,
    int nsize,
    int *core_halo_index,
    int *vsize,
    int double_jump_flag);

// Create contracted submatrix from a set of element indices.
void bml_matrix2submatrix(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int *core_halo_index,
    int lsize);

// Assemble a contracted submatrix into the final matrix.
void bml_submatrix2matrix(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int *core_halo_index,
    int lsize,
    int llsize,
    double threshold);

// Return adjacency based on rows
void bml_adjacency(
    bml_matrix_t * A,
    int *xadj,
    int *adjncy,
    int base_flag);

// Return adjacency based on groups of rows (ex. atom)
void bml_adjacency_group(
    bml_matrix_t * A,
    int *hindex,
    int nnodes,
    int *xadj,
    int *adjncy,
    int base_flag);

// Return a group-based matrix from a matrix
bml_matrix_t *bml_group_matrix(
    bml_matrix_t * A,
    int *hindex,
    int ngroups,
    double threshold);

// Return a submatrix from a matrix
bml_matrix_t *bml_extract_submatrix(
    bml_matrix_t * A,
    int irow,
    int icol,
    int B_N,
    int B_M);

void bml_assign_submatrix(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int irow,
    int icol);
#endif
