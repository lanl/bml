#ifndef __BML_THRESHOLD_ELLSORT_H
#define __BML_THRESHOLD_ELLSORT_H

#include "bml_types_ellsort.h"

bml_matrix_ellsort_t *bml_threshold_new_ellsort(
    const bml_matrix_ellsort_t * A,
    const double threshold);

bml_matrix_ellsort_t *bml_threshold_new_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const double threshold);

bml_matrix_ellsort_t *bml_threshold_new_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const double threshold);

bml_matrix_ellsort_t *bml_threshold_new_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const double threshold);

bml_matrix_ellsort_t *bml_threshold_new_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const double threshold);

void bml_threshold_ellsort(
    const bml_matrix_ellsort_t * A,
    const double threshold);

void bml_threshold_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const double threshold);

void bml_threshold_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const double threshold);

void bml_threshold_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const double threshold);

void bml_threshold_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const double threshold);

#endif
