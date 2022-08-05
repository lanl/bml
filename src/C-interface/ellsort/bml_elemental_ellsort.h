#ifndef __BML_ELEMENTAL_ELLSORT_H
#define __BML_ELEMENTAL_ELLSORT_H

#include "bml_types_ellsort.h"

#ifdef BML_COMPLEX
#include <complex.h>
#endif

float bml_get_element_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    int i,
    int j);

double bml_get_element_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    int i,
    int j);

#ifdef BML_COMPLEX
float complex bml_get_element_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    int i,
    int j);

double complex bml_get_element_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    int i,
    int j);
#endif

#endif
