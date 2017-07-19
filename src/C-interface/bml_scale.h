/** \file */

#ifndef __BML_SCALE_H
#define __BML_SCALE_H

#include "bml_types.h"

bml_matrix_t *bml_scale_new(
    const void *scale_factor,
    const bml_matrix_t * A);

void bml_scale(
    const void *scale_factor,
    const bml_matrix_t * A,
    bml_matrix_t * B);

void bml_scale_inplace(
    const void *scale_factor,
    bml_matrix_t * A);

#endif
