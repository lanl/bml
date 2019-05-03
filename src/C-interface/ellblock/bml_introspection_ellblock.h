#ifndef __BML_INTROSPECTION_ELLBLOCK_H
#define __BML_INTROSPECTION_ELLBLOCK_H

#include "bml_types_ellblock.h"

bml_matrix_precision_t bml_get_precision_ellblock(
    const bml_matrix_ellblock_t * A);

bml_distribution_mode_t bml_get_distribution_mode_ellblock(
    const bml_matrix_ellblock_t * A);

int bml_get_N_ellblock(
    const bml_matrix_ellblock_t * A);

int bml_get_M_ellblock(
    const bml_matrix_ellblock_t * A);

int bml_get_NB_ellblock(
    const bml_matrix_ellblock_t * A);

int bml_get_row_bandwidth_ellblock(
    const bml_matrix_ellblock_t * A,
    const int i);

int bml_get_bandwidth_ellblock(
    const bml_matrix_ellblock_t * A);

    // Get the sparsity of a bml matrix
double bml_get_sparsity_ellblock(
    const bml_matrix_ellblock_t * A,
    const double threshold);

double bml_get_sparsity_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const double threshold);

double bml_get_sparsity_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const double threshold);

double bml_get_sparsity_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const double threshold);

double bml_get_sparsity_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const double threshold);

#endif
