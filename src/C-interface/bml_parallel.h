/** \file */

#ifndef __BML_PARALLEL_H
#define __BML_PARALLEL_H

#include "bml_types.h"

#ifdef DO_MPI
#ifdef SINGLE
#define REAL_MPI_TYPE MPI_FLOAT
#else
#define REAL_MPI_TYPE MPI_DOUBLE
#endif
#endif

#ifdef DO_MPI
#include <mpi.h>
extern MPI_Comm ccomm;
#endif

// Return total number of processors.
int bml_getNRanks(
    void);

// Return local rank.
int bml_getMyRank(
    void);

// Initialize from Fortran MPI
void bml_initParallelF(
    int fcomm);

// Shutdown from Fortran MPI
void bml_shutdownParallelF(
    );

// Return non-zero if printing occurs from this rank.
int bml_printRank(
    void);

#ifdef DO_MPI
// Initialize some MPI stuff
void bml_initParallel(
    MPI_Comm comm);
#endif

// Deallocate some MPI stuff
void bml_shutdownParallel(
    void);

// Wrapper for MPI_Barrier.
void bml_barrierParallel(
    void);

// Wrapper for real sum MPI_AllReduce
void bml_sumRealReduce(
    double *value);

// Wrapper for real min MPI_AllReduce
void bml_minRealReduce(
    double *value);

// Wrapper for real max MPI_AllReduce
void bml_maxRealReduce(
    double *value);

// Wrapper for MPI_allGatherV
void bml_allGatherVParallel(
    bml_matrix_t * A);

#ifdef DO_MPI
void bml_mpi_send(
    bml_matrix_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_recv(
    bml_matrix_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_irecv(
    bml_matrix_t * A,
    const int dst,
    MPI_Comm comm);
void bml_mpi_irecv_complete(
    bml_matrix_t * A);
bml_matrix_t *bml_mpi_recv_matrix(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm);

void bml_mpi_bcast_matrix(
    bml_matrix_t * A,
    const int root,
    MPI_Comm comm);
#endif

#endif
