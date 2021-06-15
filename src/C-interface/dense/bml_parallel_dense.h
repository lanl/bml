#ifndef __BML_PARALLEL_DENSE_H
#define __BML_PARALLEL_DENSE_H

#include "bml_types_dense.h"

void bml_allGatherVParallel_dense(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_single_real(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_double_real(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_single_complex(
    bml_matrix_dense_t * A);

void bml_allGatherVParallel_dense_double_complex(
    bml_matrix_dense_t * A);

#ifdef DO_MPI
void bml_mpi_type_create_struct_dense(
    bml_matrix_dense_t * A,
    MPI_Datatype * newtype);

void bml_mpi_type_create_struct_dense_single_real(
    bml_matrix_dense_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_dense_double_real(
    bml_matrix_dense_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_dense_single_complex(
    bml_matrix_dense_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_dense_double_complex(
    bml_matrix_dense_t * A,
    MPI_Datatype * newtype);

void bml_mpi_send_dense(
    bml_matrix_dense_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_send_dense_single_real(
    bml_matrix_dense_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_dense_double_real(
    bml_matrix_dense_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_dense_single_complex(
    bml_matrix_dense_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_dense_double_complex(
    bml_matrix_dense_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_recv_dense(
    bml_matrix_dense_t * A,
    const int src,
    MPI_Comm comm);

void bml_mpi_recv_dense_single_real(
    bml_matrix_dense_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_dense_double_real(
    bml_matrix_dense_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_dense_single_complex(
    bml_matrix_dense_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_dense_double_complex(
    bml_matrix_dense_t * A,
    const int src,
    MPI_Comm comm);

bml_matrix_dense_t *bml_mpi_recv_matrix_dense(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm);

bml_matrix_dense_t *bml_mpi_recv_matrix_dense_single_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_dense_t *bml_mpi_recv_matrix_dense_double_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_dense_t *bml_mpi_recv_matrix_dense_single_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_dense_t *bml_mpi_recv_matrix_dense_double_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);

void bml_mpi_bcast_matrix_dense(
    bml_matrix_dense_t * A,
    const int root,
    MPI_Comm comm);
#endif

#endif
