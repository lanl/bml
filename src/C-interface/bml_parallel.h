/** \file */

#ifndef __BML_PARALLEL_H
#define __BML_PARALLEL_H

#ifdef DO_MPI
#include <mpi.h>
#endif

#include "bml_types.h"

// Return total number of processors.
int bml_getNRanks(void);

// Return local rank.
int bml_getMyRank(void);

// Return Fortran Comm
#ifdef DO_MPI
MPI_Fint bml_getComm(void);
#endif

// Return non-zero if printing occurs from this rank.
int bml_printRank(void);

// Wrapper for MPI_Init.
void bml_initParallel(int *argc, char ***argv);

// Wrapper for MPI_Finalize.
void bml_shutdownParallel(void);

// Wrapper for MPI_Barrier(MPI_COMM_WORLD).
void bml_barrierParallel(void);

#endif
