#ifndef __BML_TRACE_ELLSORT_H
#define __BML_TRACE_ELLSORT_H

#include "bml_types_ellsort.h"

double bml_trace_ellsort(
    const bml_matrix_ellsort_t * A);

double bml_trace_ellsort_single_real(
    const bml_matrix_ellsort_t * A);

double bml_trace_ellsort_double_real(
    const bml_matrix_ellsort_t * A);

double bml_trace_ellsort_single_complex(
    const bml_matrix_ellsort_t * A);

double bml_trace_ellsort_double_complex(
    const bml_matrix_ellsort_t * A);

double bml_traceMult_ellsort(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const bml_matrix_ellsort_t * B);

#endif
