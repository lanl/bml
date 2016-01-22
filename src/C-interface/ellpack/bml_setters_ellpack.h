/** \file */

#ifndef __BML_SETTERS_ELLPACK_H
#define __BML_SETTERS_ELLPACK_H

#include "bml_types_ellpack.h"

#include <complex.h>

void bml_set_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row_ellpack(
    bml_matrix_ellpack_t * A,
    const int i,
    const void *row);

void bml_set_row_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const float *row);

void bml_set_row_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int i,
    const double *row);

void bml_set_row_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const float complex * row);

void bml_set_row_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int i,
    const double complex * row);

/* Setters for diagonal */
void bml_set_diag_ellpack(
    bml_matrix_ellpack_t * A,
    const void *diag);

void bml_set_diag_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const float *diag);

void bml_set_diag_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const double *diag);

void bml_set_diag_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const float complex * diag);

void bml_set_diag_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const double complex * diag);

#endif
