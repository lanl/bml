#ifndef __BML_TRACE_DENSE_H
#define __BML_TRACE_DENSE_H

#include "bml_types_dense.h"

double bml_trace_dense(
    const bml_matrix_dense_t * A);

double bml_trace_dense_single_real(
    const bml_matrix_dense_t * A);

double bml_trace_dense_double_real(
    const bml_matrix_dense_t * A);

double bml_trace_dense_single_complex(
    const bml_matrix_dense_t * A);

double bml_trace_dense_double_complex(
    const bml_matrix_dense_t * A);

#endif
