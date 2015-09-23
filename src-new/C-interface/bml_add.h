/** \file */

#ifndef __BML_ADD_H
#define __BML_ADD_H

#include "bml_types.h"

// Matrix addition - A = alpha * A + beta * B
void bml_add(const bml_matrix_t *A, const bml_matrix_t *B, const double alpha, const double beta, const double threshold);

// Matrix addition - A = A + beta * I
void bml_add_identity(const bml_matrix_t *A, const double beta, const double threshold);

#endif
