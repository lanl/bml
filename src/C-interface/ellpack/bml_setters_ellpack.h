/** \file */

#ifndef __BML_SETTERS_DENSE_H
#define __BML_SETTERS_DENSE_H

#include "bml_types_ellpack.h"

#include <complex.h>

void bml_set_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_row_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const float *row,
    const double threshold);

void bml_set_row_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const double *row,
    const double threshold);

void bml_set_row_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const float complex * row,
    const double threshold);

void bml_set_row_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const double complex * row,
    const double threshold);

#endif
