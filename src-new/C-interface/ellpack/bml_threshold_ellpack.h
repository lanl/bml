#ifndef __BML_THRESHOLD_ELLPACK_H
#define __BML_THRESHOLD_ELLPACK_H

#include "bml_types_ellpack.h"

bml_matrix_ellpack_t *bml_threshold_new_ellpack(
    const bml_matrix_ellpack_t * A, const double threshold);

bml_matrix_ellpack_t *bml_threshold_new_ellpack_single_real(
    const bml_matrix_ellpack_t * A, const double threshold);

bml_matrix_ellpack_t *bml_threshold_new_ellpack_double_real(
    const bml_matrix_ellpack_t * A, const double threshold);

void bml_threshold_ellpack(
    const bml_matrix_ellpack_t * A, const double threshold);

void bml_threshold_ellpack_single_real(
    const bml_matrix_ellpack_t * A, const double threshold);

void bml_threshold_ellpack_double_real(
    const bml_matrix_ellpack_t * A, const double threshold);

#endif
