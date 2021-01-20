#ifndef __BML_PARALLEL_ELLSORT_H
#define __BML_PARALLEL_ELLSORT_H

#include "bml_types_ellsort.h"

void bml_allGatherVParallel_ellsort(
    bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_single_real(
    bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_double_real(
    bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_single_complex(
    bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_double_complex(
    bml_matrix_ellsort_t * A);

#ifdef DO_MPI
void bml_mpi_type_create_struct_ellsort(
    bml_matrix_ellsort_t * A,
    MPI_Datatype * newtype);

void bml_mpi_type_create_struct_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    MPI_Datatype * newtype);

void bml_mpi_send_ellsort(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_send_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_recv_ellsort(
    bml_matrix_ellsort_t * A,
    const int src,
    MPI_Comm comm);

void bml_mpi_recv_ellsort_single_real(
    bml_matrix_ellsort_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellsort_double_real(
    bml_matrix_ellsort_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellsort_single_complex(
    bml_matrix_ellsort_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellsort_double_complex(
    bml_matrix_ellsort_t * A,
    const int src,
    MPI_Comm comm);

bml_matrix_ellsort_t *bml_mpi_recv_matrix_ellsort(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm);

bml_matrix_ellsort_t *bml_mpi_recv_matrix_ellsort_single_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellsort_t *bml_mpi_recv_matrix_ellsort_double_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellsort_t *bml_mpi_recv_matrix_ellsort_single_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellsort_t *bml_mpi_recv_matrix_ellsort_double_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
#endif

#endif
