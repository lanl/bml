#ifndef __BML_TRANSPOSE_ELLSORT_H
#define __BML_TRANSPOSE_ELLSORT_H

#include "bml_types_ellsort.h"

bml_matrix_ellsort_t *bml_transpose_new_ellsort(
    const bml_matrix_ellsort_t * A);

bml_matrix_ellsort_t *bml_transpose_new_ellsort_single_real(
    const bml_matrix_ellsort_t * A);

bml_matrix_ellsort_t *bml_transpose_new_ellsort_double_real(
    const bml_matrix_ellsort_t * A);

bml_matrix_ellsort_t *bml_transpose_new_ellsort_single_complex(
    const bml_matrix_ellsort_t * A);

bml_matrix_ellsort_t *bml_transpose_new_ellsort_double_complex(
    const bml_matrix_ellsort_t * A);

void bml_transpose_ellsort(
    const bml_matrix_ellsort_t * A);

void bml_transpose_ellsort_single_real(
    const bml_matrix_ellsort_t * A);

void bml_transpose_ellsort_double_real(
    const bml_matrix_ellsort_t * A);

void bml_transpose_ellsort_single_complex(
    const bml_matrix_ellsort_t * A);

void bml_transpose_ellsort_double_complex(
    const bml_matrix_ellsort_t * A);

#endif
