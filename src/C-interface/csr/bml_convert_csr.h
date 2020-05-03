/** \file */

#ifndef __BML_CONVERT_CSR_H
#define __BML_CONVERT_CSR_H

#include "bml_types_csr.h"

bml_matrix_csr_t *bml_convert_csr(
    bml_matrix_t * A,
    bml_matrix_precision_t matrix_precision,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_csr_t *bml_convert_csr_single_real(
    bml_matrix_t * A,
    bml_matrix_precision_t matrix_precision,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_csr_t *bml_convert_csr_double_real(
    bml_matrix_t * A,
    bml_matrix_precision_t matrix_precision,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_csr_t *bml_convert_csr_single_complex(
    bml_matrix_t * A,
    bml_matrix_precision_t matrix_precision,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_csr_t *bml_convert_csr_double_complex(
    bml_matrix_t * A,
    bml_matrix_precision_t matrix_precision,
    int M,
    bml_distribution_mode_t distrib_mode);

#endif
