#ifndef __BML_ADD_DISTRIBUTED2D_H
#define __BML_ADD_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

void bml_add_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double beta,
    double threshold);

double bml_add_norm_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B,
    double alpha,
    double beta,
    double threshold);

void bml_scale_add_identity_distributed2d(
    bml_matrix_distributed2d_t * A,
    double alpha,
    double beta,
    double threshold);

void bml_add_identity_distributed2d(
    bml_matrix_distributed2d_t * A,
    double beta,
    double threshold);

#endif
