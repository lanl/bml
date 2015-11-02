#ifndef __BML_TRACE_ELLPACK_H
#define __BML_TRACE_ELLPACK_H

#include "bml_types_ellpack.h"

double bml_trace_ellpack(
    const bml_matrix_ellpack_t * A);

double bml_trace_ellpack_single_real(
    const bml_matrix_ellpack_t * A);

double bml_trace_ellpack_double_real(
    const bml_matrix_ellpack_t * A);

double bml_trace_ellpack_single_complex(
    const bml_matrix_ellpack_t * A);

double bml_trace_ellpack_double_complex(
    const bml_matrix_ellpack_t * A);

#endif
