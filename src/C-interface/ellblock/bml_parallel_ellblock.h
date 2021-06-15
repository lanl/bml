#ifndef __BML_PARALLEL_ELLBLOCK_H
#define __BML_PARALLEL_ELLBLOCK_H

#include "bml_types_ellblock.h"

#ifdef DO_MPI
void bml_mpi_send_ellblock(
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_send_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_send_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm);

void bml_mpi_recv_ellblock(
    bml_matrix_ellblock_t * A,
    const int src,
    MPI_Comm comm);

void bml_mpi_recv_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const int src,
    MPI_Comm comm);
void bml_mpi_recv_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const int src,
    MPI_Comm comm);

bml_matrix_ellblock_t *bml_mpi_recv_matrix_ellblock(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm);

bml_matrix_ellblock_t *bml_mpi_recv_matrix_ellblock_single_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellblock_t *bml_mpi_recv_matrix_ellblock_double_real(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellblock_t *bml_mpi_recv_matrix_ellblock_single_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);
bml_matrix_ellblock_t *bml_mpi_recv_matrix_ellblock_double_complex(
    int N,
    int M,
    const int src,
    MPI_Comm comm);

void bml_mpi_bcast_matrix_ellblock(
    bml_matrix_ellblock_t * A,
    const int root,
    MPI_Comm comm);

void bml_mpi_bcast_matrix_ellblock_single_real(
    bml_matrix_ellblock_t * A,
    const int root,
    MPI_Comm comm);

void bml_mpi_bcast_matrix_ellblock_double_real(
    bml_matrix_ellblock_t * A,
    const int root,
    MPI_Comm comm);

void bml_mpi_bcast_matrix_ellblock_single_complex(
    bml_matrix_ellblock_t * A,
    const int root,
    MPI_Comm comm);

void bml_mpi_bcast_matrix_ellblock_double_complex(
    bml_matrix_ellblock_t * A,
    const int root,
    MPI_Comm comm);
#endif

#endif
