#ifndef __BML_TYPES_DISTRIBUTED_H
#define __BML_TYPES_DISTRIBUTED_H

#include "../bml_types.h"
#include <mpi.h>

/** Distributed matrix type. */
struct bml_matrix_distributed2d_t
{
    /** The matrix type identifier. */
    bml_matrix_type_t matrix_type;
    /** The real precision. */
    bml_matrix_precision_t matrix_precision;
    /** The number of rows/columns. */
    int N;
    /** The local submatrix */
    bml_matrix_t *matrix;
    /** local submatrix dimensions */
    int n;
    /** MPI communicator */
    MPI_Comm comm;
    /** number of MPI tasks */
    int ntasks;
    /** number of processors rows */
    int nprows;
    /** number of processors cols  */
    int npcols;
    int myprow;
    int mypcol;
    /** local MPI task ID */
    int mpitask;
};
typedef struct bml_matrix_distributed2d_t bml_matrix_distributed2d_t;

#endif
