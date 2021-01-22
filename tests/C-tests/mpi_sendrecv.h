#ifndef __TEST_MPI_SENDRECV_H
#define __TEST_MPI_SENDRECV_H

#include <bml.h>

int test_mpi_sendrecv(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_mpi_sendrecv_single_real(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_mpi_sendrecv_double_real(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_mpi_sendrecv_single_complex(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

int test_mpi_sendrecv_double_complex(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

#endif
