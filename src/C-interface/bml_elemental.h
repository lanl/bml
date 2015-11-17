#ifndef __BML_ELEMENTAL
#define __BML_ELEMENTAL

#include "bml_types.h"

#include <complex.h>

float bml_get_single_real(
    const bml_matrix_t * A,
    const int i,
    const int j);

double bml_get_double_real(
    const bml_matrix_t * A,
    const int i,
    const int j);

float complex bml_get_single_complex(
    const bml_matrix_t * A,
    const int i,
    const int j);

double complex bml_get_double_complex(
    const bml_matrix_t * A,
    const int i,
    const int j);

#endif
