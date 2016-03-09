/** \file */

#ifndef __BML_SUBMATRIX_H
#define __BML_SUBMATRIX_H

#include "bml_types.h"

// Determine element indeces for submatrix, given a set of nodes.
void bml_matrix2submatrix_index(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const int * nodelist,
    const int nsize,
    int * core_halo_index,
    int * core_pos,
    int * vsize,
    const int double_jump_flag);

// Create contracted submatrix from a set of element indeces.
void bml_matrix2submatrix(
    const bml_matrix_t * A,
    bml_matrix_t * B,
    const int * core_halo_index,
    const int lsize);

// Assemble a contracted submatrix into the final matrix.
void bml_submatrix2matrix(
    const bml_matrix_t * A,
    bml_matrix_t * B,
    const int * core_halo_index,
    const int lsize,
    const int * core_pos,
    const int llsize,
    const double threshold);
    
#endif
