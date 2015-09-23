/** \file */

#ifndef __BML_UTILITIES_H
#define __BML_UTILITIES_H

#include "bml_types.h"

void bml_print_matrix(
    const int N,
    bml_matrix_precision_t matrix_precision,
    const void *A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

#endif
