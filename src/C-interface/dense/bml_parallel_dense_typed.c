#include "../../macros.h"
#include "../../typed.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_allocate.h"
#include "bml_parallel_dense.h"
#include "bml_types_dense.h"
#include "bml_allocate_dense.h"
#include "../bml_logger.h"

#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#endif

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Gather a bml matrix across MPI ranks.
 *
 *  \ingroup parallel_group
 *
 *  \param A The matrix
 */
void TYPED_FUNC(
    bml_allGatherVParallel_dense) (
    bml_matrix_dense_t * A)
{
#ifdef DO_MPI
    int myRank = bml_getMyRank();

    int N = A->N;

    REAL_T *A_matrix = A->matrix;

    bml_domain_t *A_domain = A->domain;

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   A_matrix, A_domain->localElements,
                   A_domain->localDispl, REAL_MPI_TYPE, ccomm);
#endif
}

#ifdef DO_MPI

void TYPED_FUNC(
    bml_mpi_type_create_struct_dense) (
    bml_matrix_dense_t * A,
    MPI_Datatype * newtype)
{
    assert(A->N > 0);

    MPI_Aint baseaddr;
    MPI_Aint addr0;
    MPI_Get_address(A, &baseaddr);
    MPI_Get_address(A->matrix, &addr0);

    MPI_Datatype dtype[1];
    dtype[0] = MPI_T;

    int blength[1];
    blength[0] = A->N * A->N;
    MPI_Aint displ[0];
    displ[0] = addr0 - baseaddr;
    int mpiret = MPI_Type_create_struct(1, blength, displ, dtype, newtype);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Type_create_struct failed!");
    mpiret = MPI_Type_commit(newtype);
    if (mpiret != MPI_SUCCESS)
        LOG_ERROR("MPI_Type_commit failed!");
}

void TYPED_FUNC(
    bml_mpi_send_dense) (
    bml_matrix_dense_t * A,
    const int dst,
    MPI_Comm comm)
{
#ifdef BML_USE_MAGMA
    MAGMA_T *A_matrix = bml_allocate_memory(sizeof(MAGMA_T) * A->N * A->N);
    MAGMA(getmatrix) (A->N, A->N, A->matrix, A->ld, A_matrix, A->N,
                      bml_queue());
#else
    REAL_T *A_matrix = A->matrix;
#endif

    MPI_Send(A_matrix, A->N * A->N, MPI_T, dst, 222, comm);

#ifdef BML_USE_MAGMA
    free(A_matrix);
#endif
}

void TYPED_FUNC(
    bml_mpi_recv_dense) (
    bml_matrix_dense_t * A,
    const int src,
    MPI_Comm comm)
{
#ifdef BML_USE_MAGMA
    MAGMA_T *A_matrix = bml_allocate_memory(sizeof(MAGMA_T) * A->N * A->N);
#else
    REAL_T *A_matrix = A->matrix;
#endif

    MPI_Status status;
    MPI_Recv(A_matrix, A->N * A->N, MPI_T, src, 222, comm, &status);

#ifdef BML_USE_MAGMA
    MAGMA(setmatrix) (A->N, A->N, A_matrix, A->N, A->matrix, A->ld,
                      bml_queue());
    free(A_matrix);
#endif
}

bml_matrix_dense_t
    * TYPED_FUNC(bml_mpi_recv_matrix_dense) (int N, int M,
                                             const int src, MPI_Comm comm)
{
    bml_matrix_dimension_t matrix_dimension = { N, N, M };
    bml_matrix_dense_t *A_bml =
        TYPED_FUNC(bml_zero_matrix_dense) (matrix_dimension, sequential);

    bml_mpi_recv_dense(A_bml, src, comm);

    return A_bml;
}

#endif
