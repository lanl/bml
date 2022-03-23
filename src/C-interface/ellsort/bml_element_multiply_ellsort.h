#ifndef __BML_ELEMNET_MULTIPLY_ELLSORT_H
#define __BML_ELEMNET_MULTIPLY_ELLSORT_H

#include "bml_types_ellsort.h"

void bml_element_multiply_AB_ellsort(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    double threshold);

void bml_element_multiply_AB_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    double threshold);

void bml_element_multiply_AB_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    double threshold);

void bml_element_multiply_AB_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    double threshold);

void bml_element_multiply_AB_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
    double threshold);

#endif
