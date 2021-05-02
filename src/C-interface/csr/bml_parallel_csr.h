#ifndef __BML_PARALLEL_CSR_H
#define __BML_PARALLEL_CSR_H

#include "bml_types_csr.h"

#ifdef DO_MPI
void bml_mpi_send_csr(
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_send_csr_single_real(
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_csr_double_real(
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_csr_single_complex(
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_csr_double_complex(
    bml_matrix_csr_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_recv_csr(
    bml_matrix_csr_t * A,
    const int src,
    MPI_Comm comm);

void bml_mpi_recv_csr_single_real(
    bml_matrix_csr_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_csr_double_real(
    bml_matrix_csr_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_csr_single_complex(
    bml_matrix_csr_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_csr_double_complex(
    bml_matrix_csr_t * A,
    const int src,
    MPI_Comm comm);

bml_matrix_csr_t *bml_mpi_recv_matrix_csr(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm);

bml_matrix_csr_t *bml_mpi_recv_matrix_csr_single_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_csr_t *bml_mpi_recv_matrix_csr_double_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_csr_t *bml_mpi_recv_matrix_csr_single_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_csr_t *bml_mpi_recv_matrix_csr_double_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
#endif

#endif
