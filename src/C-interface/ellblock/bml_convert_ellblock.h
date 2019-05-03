/** \file */

#ifndef __BML_CONVERT_ELLBLOCK_H
#define __BML_CONVERT_ELLBLOCK_H

#include "bml_types_ellblock.h"

bml_matrix_ellblock_t *bml_convert_ellblock(
    const bml_matrix_t * A,
    const bml_matrix_precision_t matrix_precision,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_convert_ellblock_single_real(
    const bml_matrix_t * A,
    const bml_matrix_precision_t matrix_precision,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_convert_ellblock_double_real(
    const bml_matrix_t * A,
    const bml_matrix_precision_t matrix_precision,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_convert_ellblock_single_complex(
    const bml_matrix_t * A,
    const bml_matrix_precision_t matrix_precision,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_convert_ellblock_double_complex(
    const bml_matrix_t * A,
    const bml_matrix_precision_t matrix_precision,
    const int M,
    const bml_distribution_mode_t distrib_mode);

#endif
