#ifndef __BML_TRANSPOSE_TRIANGLE_DENSE_H
#define __BML_TRANSPOSE_TRIANGLE_DENSE_H

#include "bml_types_dense.h"

void bml_transpose_triangle_dense(
    bml_matrix_dense_t * A,
    char triangle);

void bml_transpose_triangle_dense_single_real(
    bml_matrix_dense_t * A,
    char triangle);

void bml_transpose_triangle_dense_double_real(
    bml_matrix_dense_t * A,
    char triangle);

void bml_transpose_triangle_dense_single_complex(
    bml_matrix_dense_t * A,
    char triangle);

void bml_transpose_triangle_dense_double_complex(
    bml_matrix_dense_t * A,
    char triangle);

#endif
