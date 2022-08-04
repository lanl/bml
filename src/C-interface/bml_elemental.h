#ifndef __BML_ELEMENTAL
#define __BML_ELEMENTAL

#include "bml_types.h"

#ifdef BML_COMPLEX
#include <complex.h>
#endif

float bml_get_element_single_real(
    bml_matrix_t * A,
    int i,
    int j);

double bml_get_element_double_real(
    bml_matrix_t * A,
    int i,
    int j);

#ifdef BML_COMPLEX
float _Complex bml_get_element_single_complex(
    bml_matrix_t * A,
    int i,
    int j);

double _Complex bml_get_element_double_complex(
    bml_matrix_t * A,
    int i,
    int j);
#endif

#endif
