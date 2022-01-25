#ifndef __ELEMENT_MULTIPLY_MATRIX_H
#define __ELEMENT_MULTIPLY_MATRIX_H

#include <bml.h>

int test_element_multiply(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_element_multiply_single_real(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_element_multiply_double_real(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_element_multiply_single_complex(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_element_multiply_double_complex(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

#endif
