#ifndef __BML_TRACE_CSR_H
#define __BML_TRACE_CSR_H

#include "bml_types_csr.h"

double bml_trace_csr(
    const bml_matrix_csr_t * A);

double bml_trace_csr_single_real(
    const bml_matrix_csr_t * A);

double bml_trace_csr_double_real(
    const bml_matrix_csr_t * A);

double bml_trace_csr_single_complex(
    const bml_matrix_csr_t * A);

double bml_trace_csr_double_complex(
    const bml_matrix_csr_t * A);

double bml_trace_mult_csr(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_trace_mult_csr_single_real(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_trace_mult_csr_double_real(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_trace_mult_csr_single_complex(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_trace_mult_csr_double_complex(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

#endif
