#ifndef __BML_TRACE_ELLBLOCK_H
#define __BML_TRACE_ELLNLOCK_H

#include "bml_types_ellblock.h"

double bml_trace_ellblock(
    const bml_matrix_ellblock_t * A);

double bml_trace_ellblock_single_real(
    const bml_matrix_ellblock_t * A);

double bml_trace_ellblock_double_real(
    const bml_matrix_ellblock_t * A);

double bml_trace_ellblock_single_complex(
    const bml_matrix_ellblock_t * A);

double bml_trace_ellblock_double_complex(
    const bml_matrix_ellblock_t * A);

double bml_traceMult_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_traceMult_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_traceMult_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_traceMult_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

double bml_traceMult_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B);

#endif
