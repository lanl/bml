/** \file */

#ifndef __BML_SCALE_H
#define __BML_SCALE_H

#include "bml_types.h"

// Scales A and returns a new B
bml_matrix_t *bml_scale_new(
    const double scale_factor,
    const bml_matrix_t * A);

// Scales A and returns in existing B
void bml_scale(
    const double scale_factor,
    const bml_matrix_t * A,
    bml_matrix_t * B);

void bml_scale_inplace(
    const double scale_factor,
    bml_matrix_t * A);

#endif
