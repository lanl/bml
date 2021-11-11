#ifndef __BML_INTROSPECTION_DISTRIBUTED2D_H
#define __BML_INTROSPECTION_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

bml_matrix_precision_t bml_get_precision_distributed2d(
    bml_matrix_distributed2d_t * A);

bml_distribution_mode_t bml_get_distribution_mode_distributed2d(
    bml_matrix_distributed2d_t * A);

int bml_get_N_distributed2d(
    bml_matrix_distributed2d_t * A);

int bml_get_M_distributed2d(
    bml_matrix_distributed2d_t * A);

bml_matrix_t *bml_get_local_matrix_distributed2d(
    bml_matrix_distributed2d_t * A);

double bml_get_sparsity_distributed2d(
    bml_matrix_distributed2d_t * A,
    double threshold);
#endif
