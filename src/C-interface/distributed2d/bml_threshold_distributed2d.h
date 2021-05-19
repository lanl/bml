#ifndef __BML_THRESHOLD_DISTRIBUTED2D_H
#define __BML_THRESHOLD_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

bml_matrix_distributed2d_t *bml_threshold_new_distributed2d(
    bml_matrix_distributed2d_t * A,
    double threshold);

bml_matrix_distributed2d_t
    * bml_threshold_new_distributed2d_single_real(bml_matrix_distributed2d_t *
                                                  A, double threshold);

bml_matrix_distributed2d_t
    * bml_threshold_new_distributed2d_double_real(bml_matrix_distributed2d_t *
                                                  A, double threshold);

bml_matrix_distributed2d_t
    *
bml_threshold_new_distributed2d_single_complex(bml_matrix_distributed2d_t * A,
                                               double threshold);

bml_matrix_distributed2d_t
    *
bml_threshold_new_distributed2d_double_complex(bml_matrix_distributed2d_t * A,
                                               double threshold);

#endif
