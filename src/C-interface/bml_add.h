/** \file */

#ifndef __BML_ADD_H
#define __BML_ADD_H

#include "bml_types.h"

void bml_add(
    bml_matrix_t * A,
    const bml_matrix_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_add_norm(
    bml_matrix_t * A,
    const bml_matrix_t * B,
    const double alpha,
    const double beta,
    const double threshold);

void bml_add_identity(
    bml_matrix_t * A,
    const double beta,
    const double threshold);

#endif
