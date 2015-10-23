/** \file */

#ifndef __BML_COPY_H
#define __BML_COPY_H

#include "bml_types.h"

// Copy A to a new matrix
bml_matrix_t *bml_copy_new(
    const bml_matrix_t * A);

// Copy A to an existing matrix B
void bml_copy(
    const bml_matrix_t * A,
    bml_matrix_t ** B);

#endif
