#ifndef __BML_NORMALIZE_DISTRIBUTED2D_H
#define __BML_NORMALIZE_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

void bml_normalize_distributed2d(
    bml_matrix_distributed2d_t * A,
    double mineval,
    double maxeval);

void bml_normalize_distributed2d_single_real(
    bml_matrix_distributed2d_t * A,
    double mineval,
    double maxeval);

void bml_normalize_distributed2d_double_real(
    bml_matrix_distributed2d_t * A,
    double mineval,
    double maxeval);

void bml_normalize_distributed2d_single_complex(
    bml_matrix_distributed2d_t * A,
    double mineval,
    double maxeval);

void bml_normalize_distributed2d_double_complex(
    bml_matrix_distributed2d_t * A,
    double mineval,
    double maxeval);

void *bml_gershgorin_distributed2d(
    bml_matrix_distributed2d_t * A);

void *bml_gershgorin_distributed2d_single_real(
    bml_matrix_distributed2d_t * A);

void *bml_gershgorin_distributed2d_double_real(
    bml_matrix_distributed2d_t * A);

void *bml_gershgorin_distributed2d_single_complex(
    bml_matrix_distributed2d_t * A);

void *bml_gershgorin_distributed2d_double_complex(
    bml_matrix_distributed2d_t * A);

#endif
