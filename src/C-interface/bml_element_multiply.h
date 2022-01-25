/** \file */

#ifndef __BML_ELEMNET_MULTIPLY_H
#define __BML_ELEMNET_MULTIPLY_H

#include "bml_types.h"

// Element-wise Matrix multiply (Hadamard product)
void bml_element_multiply_AB(
    bml_matrix_t * A,
    bml_matrix_t * B,
    bml_matrix_t * C,
    double threshold);

#endif
