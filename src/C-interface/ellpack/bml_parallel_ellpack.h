#ifndef __BML_PARALLEL_ELLPACK_H
#define __BML_PARALLEL_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_allGatherVParallel_ellpack(
    bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_single_real(
    bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_double_real(
    bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_single_complex(
    bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_double_complex(
    bml_matrix_ellpack_t * A);

#ifdef DO_MPI
void bml_mpi_type_create_struct_ellpack(
    bml_matrix_ellpack_t * A,
    MPI_Datatype * newtype);

void bml_mpi_type_create_struct_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    MPI_Datatype * newtype);
void bml_mpi_type_create_struct_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    MPI_Datatype * newtype);

void bml_mpi_send_ellpack(
    bml_matrix_ellpack_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_send_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_recv_ellpack(
    bml_matrix_ellpack_t * A,
    const int src,
    MPI_Comm comm);

void bml_mpi_recv_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    const int src,
    MPI_Comm comm);

bml_matrix_ellpack_t *bml_mpi_recv_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm);

bml_matrix_ellpack_t *bml_mpi_recv_matrix_ellpack_single_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellpack_t *bml_mpi_recv_matrix_ellpack_double_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellpack_t *bml_mpi_recv_matrix_ellpack_single_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellpack_t *bml_mpi_recv_matrix_ellpack_double_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);

void bml_mpi_bcast_matrix_ellpack(
    bml_matrix_ellpack_t * A,
    const int root,
    MPI_Comm comm);
#endif

#endif
