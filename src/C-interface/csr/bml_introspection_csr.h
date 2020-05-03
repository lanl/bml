#ifndef __BML_INTROSPECTION_CSR_H
#define __BML_INTROSPECTION_CSR_H

#include "bml_types_csr.h"

bml_matrix_precision_t bml_get_precision_csr(
    bml_matrix_csr_t * A);

bml_distribution_mode_t bml_get_distribution_mode_csr(
    bml_matrix_csr_t * A);

int bml_get_N_csr(
    bml_matrix_csr_t * A);

int bml_get_M_csr(
    bml_matrix_csr_t * A);

int bml_get_row_bandwidth_csr(
    bml_matrix_csr_t * A,
    int i);

int bml_get_bandwidth_csr(
    bml_matrix_csr_t * A);

    // Get the sparsity of a bml matrix
double bml_get_sparsity_csr(
    bml_matrix_csr_t * A,
    double threshold);

double bml_get_sparsity_csr_single_real(
    bml_matrix_csr_t * A,
    double threshold);

double bml_get_sparsity_csr_double_real(
    bml_matrix_csr_t * A,
    double threshold);

double bml_get_sparsity_csr_single_complex(
    bml_matrix_csr_t * A,
    double threshold);

double bml_get_sparsity_csr_double_complex(
    bml_matrix_csr_t * A,
    double threshold);

#endif
