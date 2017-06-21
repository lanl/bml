/** \file */

#ifndef __BML_INTROSPECTION_H
#define __BML_INTROSPECTION_H

#include "bml_types.h"

bml_matrix_type_t bml_get_type(
    const bml_matrix_t * A);

bml_matrix_precision_t bml_get_precision(
    const bml_matrix_t * A);

int bml_get_N(
    const bml_matrix_t * A);

int bml_get_M(
    const bml_matrix_t * A);

int bml_get_row_bandwidth(
    const bml_matrix_t * A,
    const int i);

int bml_get_bandwidth(
    const bml_matrix_t * A);

double bml_get_sparsity(
    const bml_matrix_t * A,
    const double threshold);

bml_distribution_mode_t bml_get_distribution_mode(
    const bml_matrix_t * A);

#endif
