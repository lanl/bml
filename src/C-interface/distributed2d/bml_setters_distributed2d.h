/** \file */

#ifndef __BML_SETTERS_DISTRIBUTED2D_H
#define __BML_SETTERS_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

void bml_set_diagonal_distributed2d(
    bml_matrix_distributed2d_t * A,
    void *diagonal,
    double threshold);

#endif
