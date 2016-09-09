/** \file */

#ifndef __BML_SUBMATRIX_H
#define __BML_SUBMATRIX_H

#include "bml_types.h"

// Determine element indeces for submatrix, given a set of nodes.
void bml_matrix2submatrix_index(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const int *nodelist,
    const int nsize,
    int *core_halo_index,
    int *vsize,
    const int double_jump_flag);

// Determine core+halo indeces from graph only
void bml_matrix2submatrix_index_graph(
    const bml_matrix_t * B,
    const int *nodelist,
    const int nsize,
    int *core_halo_index,
    int *vsize,
    const int double_jump_flag);

// Create contracted submatrix from a set of element indeces.
void bml_matrix2submatrix(
    const bml_matrix_t * A,
    bml_matrix_t * B,
    const int *core_halo_index,
    const int lsize);

// Assemble a contracted submatrix into the final matrix.
void bml_submatrix2matrix(
    const bml_matrix_t * A,
    bml_matrix_t * B,
    const int *core_halo_index,
    const int lsize,
    const int llsize,
    const double threshold);

// Return adjacency based on rows
void bml_adjacency(
	const bml_matrix_t * A,
	int * xadj,
	int * adjncy,
        const int base_flag);
	
// Return adjacency based on groups of rows (ex. atom)
void bml_adjacency_group(
        const bml_matrix_t * A,
        const int * hindex,
        const int nnodes,
        int * xadj,
        int * adjncy,
        const int base_flag);

// Return a group-based matrix from a matrix
bml_matrix_t *bml_group_matrix(
    const bml_matrix_t * A,
    const int * hindex,
    const int ngroups,
    const double threshold);

#endif
