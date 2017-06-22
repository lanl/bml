/** \file */

#ifndef __BML_SPARSITY_H
#define __BML_SPARSITY_H

#include "bml_types.h"

// Calculates the sparsity of a bml matrix
// This is NumberOfZeroes/(NxN) where N is
// the matrix dimension.
double bml_get_sparsity(
    const bml_matrix_t * A);

#endif
