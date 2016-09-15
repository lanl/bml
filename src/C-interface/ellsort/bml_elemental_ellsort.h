#ifndef __BML_ELEMENTAL_ELLSORT_H
#define __BML_ELEMENTAL_ELLSORT_H

#include "bml_types_ellsort.h"

#include <complex.h>

float bml_get_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

double bml_get_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

float complex bml_get_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

double complex bml_get_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const int i,
    const int j);

#endif
