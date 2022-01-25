#ifndef __BML_ELEMNET_MULTIPLY_CSR_H
#define __BML_ELEMNET_MULTIPLY_CSR_H

#include "bml_types_csr.h"

void bml_element_multiply_AB_csr(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    bml_matrix_csr_t * C,
    double threshold);

void bml_element_multiply_AB_csr_single_real(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    bml_matrix_csr_t * C,
    double threshold);

void bml_element_multiply_AB_csr_double_real(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    bml_matrix_csr_t * C,
    double threshold);

void bml_element_multiply_AB_csr_single_complex(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    bml_matrix_csr_t * C,
    double threshold);

void bml_element_multiply_AB_csr_double_complex(
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    bml_matrix_csr_t * C,
    double threshold);

#endif
