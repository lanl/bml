#ifndef __BML_ELEMENTAL_ELLPACK_H
#define __BML_ELEMENTAL_ELLPACK_H

#include "bml_types_ellpack.h"

#include <complex.h>

float bml_get_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    const int i,
    const int j);

double bml_get_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    const int i,
    const int j);

float complex bml_get_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    const int i,
    const int j);

double complex bml_get_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    const int i,
    const int j);

#endif
