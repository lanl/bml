#ifndef __BML_ALLOCATE_H
#define __BML_ALLOCATE_H

#include "bml_types.h"

BMLMatrix *
bml_identity_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

#endif
