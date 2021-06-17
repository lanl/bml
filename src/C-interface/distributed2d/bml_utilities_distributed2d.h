#ifndef __BML_UTILITIES_DISTRIBUTED2D_H
#define __BML_UTILITIES_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

void bml_read_bml_matrix_distributed2d(
    bml_matrix_distributed2d_t * A,
    char *filename);

void bml_write_bml_matrix_distributed2d(
    bml_matrix_distributed2d_t * A,
    char *filename);
#endif
