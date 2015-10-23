#ifndef __BML_INTROSPECTION_ELLPACK_H
#define __BML_INTROSPECTION_ELLPACK_H

#include "bml_types_ellpack.h"

bml_matrix_precision_t bml_get_precision_ellpack(
    const bml_matrix_ellpack_t * A);

int bml_get_N_ellpack(
    const bml_matrix_ellpack_t * A);

int bml_get_M_ellpack(
    const bml_matrix_ellpack_t * A);

#endif
