#ifndef __BML_CONVERT_DENSE_H
#define __BML_CONVERT_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_convert_from_dense_dense(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const void *A,
    const double threshold);

bml_matrix_dense_t *bml_convert_from_dense_dense_single_real(
    const int N,
    const void *A);

bml_matrix_dense_t *bml_convert_from_dense_dense_double_real(
    const int N,
    const void *A);

bml_matrix_dense_t *bml_convert_from_dense_dense_single_complex(
    const int N,
    const void *A);

bml_matrix_dense_t *bml_convert_from_dense_dense_double_complex(
    const int N,
    const void *A);

void *bml_convert_to_dense_dense(
    const bml_matrix_dense_t * A);

void *bml_convert_to_dense_dense_single_real(
    const bml_matrix_dense_t * A);

void *bml_convert_to_dense_dense_double_real(
    const bml_matrix_dense_t * A);

void *bml_convert_to_dense_dense_single_complex(
    const bml_matrix_dense_t * A);

void *bml_convert_to_dense_dense_double_complex(
    const bml_matrix_dense_t * A);

#endif
