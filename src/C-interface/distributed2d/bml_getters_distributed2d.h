#ifndef __BML_GETTERS_DISTRIBUTED2D_H
#define __BML_GETTERS_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

void *bml_get_row_distributed2d(
    bml_matrix_distributed2d_t * A,
    int i);

void *bml_get_row_distributed2d_single_real(
    bml_matrix_distributed2d_t * A,
    int i);

void *bml_get_row_distributed2d_double_real(
    bml_matrix_distributed2d_t * A,
    int i);

void *bml_get_row_distributed2d_single_complex(
    bml_matrix_distributed2d_t * A,
    int i);

void *bml_get_row_distributed2d_double_complex(
    bml_matrix_distributed2d_t * A,
    int i);

#endif
