#ifndef __BML_TRACE_DISTRIBUTED2D_H
#define __BML_TRACE_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

double bml_trace_distributed2d(
    bml_matrix_distributed2d_t * A);

double bml_trace_mult_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_matrix_distributed2d_t * B);

#endif
