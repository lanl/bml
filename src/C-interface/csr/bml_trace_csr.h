#ifndef __BML_TRACE_csr_H
#define __BML_TRACE_csr_H

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

double bml_traceMult_csr(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_traceMult_csr_single_real(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_traceMult_csr_double_real(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_traceMult_csr_single_complex(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

double bml_traceMult_csr_double_complex(
    const bml_matrix_csr_t * A,
    const bml_matrix_csr_t * B);

#endif
