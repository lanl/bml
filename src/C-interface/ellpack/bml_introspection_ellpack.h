#ifndef __BML_INTROSPECTION_ELLPACK_H
#define __BML_INTROSPECTION_ELLPACK_H

#include "bml_types_ellpack.h"

bml_matrix_precision_t bml_get_precision_ellpack(
    const bml_matrix_ellpack_t * A);

bml_distribution_mode_t bml_get_distribution_mode_ellpack(
    const bml_matrix_ellpack_t * A);

int bml_get_N_ellpack(
    const bml_matrix_ellpack_t * A);

int bml_get_M_ellpack(
    const bml_matrix_ellpack_t * A);

int bml_get_row_bandwidth_ellpack(
    const bml_matrix_ellpack_t * A,
    const int i);

int bml_get_bandwidth_ellpack(
    const bml_matrix_ellpack_t * A);

    // Get the sparsity of a bml matrix
double bml_get_sparsity_ellpack(
    const bml_matrix_ellpack_t * A,
    const double threshold);

double bml_get_sparsity_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    const double threshold);

double bml_get_sparsity_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    const double threshold);

double bml_get_sparsity_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    const double threshold);

double bml_get_sparsity_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    const double threshold);

#endif
