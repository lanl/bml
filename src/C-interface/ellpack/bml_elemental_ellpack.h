#ifndef __BML_ELEMENTAL_ELLPACK_H
#define __BML_ELEMENTAL_ELLPACK_H

#include "bml_types_ellpack.h"

#ifdef BML_COMPLEX
#include <complex.h>
#endif

float bml_get_element_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    int i,
    int j);

double bml_get_element_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    int i,
    int j);

#ifdef BML_COMPLEX
float _Complex bml_get_element_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    int i,
    int j);

double _Complex bml_get_element_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    int i,
    int j);
#endif

#endif
