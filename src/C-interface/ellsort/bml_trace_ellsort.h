#ifndef __BML_TRACE_ELLSORT_H
#define __BML_TRACE_ELLSORT_H

#include "bml_types_ellsort.h"

double bml_trace_ellsort(
    bml_matrix_ellsort_t * A);

double bml_trace_ellsort_single_real(
    bml_matrix_ellsort_t * A);

double bml_trace_ellsort_double_real(
    bml_matrix_ellsort_t * A);

double bml_trace_ellsort_single_complex(
    bml_matrix_ellsort_t * A);

double bml_trace_ellsort_double_complex(
    bml_matrix_ellsort_t * A);

double bml_traceMult_ellsort(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B);

double bml_traceMult_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B);

#endif
