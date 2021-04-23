#ifndef __BML_TRANSPOSE_DISTRIBUTED2D_H
#define __BML_TRANSPOSE_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

bml_matrix_distributed2d_t *bml_transpose_new_distributed2d(
    bml_matrix_distributed2d_t * A);

void bml_transpose_distributed2d(
    bml_matrix_distributed2d_t * A);

#endif
