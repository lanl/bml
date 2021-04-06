#ifndef __BML_CONVERT_DISTRIBUTED_H
#define __BML_CONVERT_DISTRIBUTED_H

#include "bml_types_distributed2d.h"

bml_matrix_distributed2d_t *bml_convert_distributed2d(
    bml_matrix_t * A,
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int M);

#endif
