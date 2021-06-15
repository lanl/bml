#include "../../macros.h"
#include "../../typed.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "../bml_allocate.h"
#include "bml_parallel_ellblock.h"
#include "bml_types_ellblock.h"
#include "bml_allocate_ellblock.h"
#include "../bml_logger.h"
#include "../bml_utilities.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef DO_MPI
#include <mpi.h>
#endif

#ifdef DO_MPI
void TYPED_FUNC(
    bml_mpi_send_ellblock) (
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm)
{
    // allocate buffers and send data
    int *indexb = bml_allocate_memory(sizeof(int) * A->NB * A->MB);
    memcpy(indexb, A->indexb, A->NB * A->MB * sizeof(int));
    MPI_Send(indexb, A->NB * A->MB, MPI_INT, dst, 112, comm);

    int *nnzb = bml_allocate_memory(sizeof(int) * A->NB);
    memcpy(nnzb, A->nnzb, sizeof(int) * A->NB);
    MPI_Send(nnzb, A->NB, MPI_INT, dst, 113, comm);

    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;

    REAL_T *values = bml_allocate_memory(sizeof(REAL_T) * A->N * A->M);
    REAL_T *pvalues = values;
    for (int ib = 0; ib < A->NB; ib++)
    {
        for (int jp = 0; jp < A->nnzb[ib]; jp++)
        {
            int ind = ROWMAJOR(ib, jp, A->NB, A->MB);
            int jb = A->indexb[ind];
            int nelements = A->bsize[ib] * A->bsize[jb];
            memcpy(pvalues, A_ptr_value[ind], nelements * sizeof(REAL_T));
            pvalues += nelements;
        }
    }
    MPI_Send(values, A->N * A->M, MPI_T, dst, 111, comm);

    bml_free_memory(nnzb);
    bml_free_memory(indexb);
    bml_free_memory(values);
}

void TYPED_FUNC(
    bml_mpi_recv_ellblock) (
    bml_matrix_ellblock_t * A,
    const int src,
    MPI_Comm comm)
{
    MPI_Status status;

    // allocate buffers and receive data
    int *indexb = bml_allocate_memory(sizeof(int) * A->NB * A->MB);
    MPI_Recv(indexb, A->NB * A->MB, MPI_INT, src, 112, comm, &status);

    int *nnzb = bml_allocate_memory(sizeof(int) * A->NB);
    MPI_Recv(nnzb, A->NB, MPI_INT, src, 113, comm, &status);

    REAL_T *values = bml_allocate_memory(sizeof(REAL_T) * A->N * A->M);
    MPI_Recv(values, A->N * A->M, MPI_T, src, 111, comm, &status);

    // copy data into bml matrix
    memcpy(A->nnzb, nnzb, sizeof(int) * A->NB);
    memcpy(A->indexb, indexb, A->NB * A->MB * sizeof(int));

    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;

    REAL_T *pvalues = values;
    for (int ib = 0; ib < A->NB; ib++)
    {
        for (int jp = 0; jp < A->nnzb[ib]; jp++)
        {
            int ind = ROWMAJOR(ib, jp, A->NB, A->MB);
            int jb = A->indexb[ind];
            int nelements = A->bsize[ib] * A->bsize[jb];
            A_ptr_value[ind] =
                TYPED_FUNC(bml_allocate_block_ellblock) (A, ib, nelements);
            memcpy(A_ptr_value[ind], pvalues, nelements * sizeof(REAL_T));
            pvalues += nelements;
        }
    }
    bml_free_memory(nnzb);
    bml_free_memory(indexb);
    bml_free_memory(values);
}

/*
 * Return BML matrix from data received from MPI task src
 */
bml_matrix_ellblock_t
    * TYPED_FUNC(bml_mpi_recv_matrix_ellblock) (int N, int M,
                                                const int src, MPI_Comm comm)
{
    bml_matrix_ellblock_t *A_bml =
        TYPED_FUNC(bml_zero_matrix_ellblock) (N, M, sequential);

    bml_mpi_recv_ellblock(A_bml, src, comm);

    return A_bml;
}

void TYPED_FUNC(
    bml_mpi_bcast_matrix_ellblock) (
    bml_matrix_ellblock_t * A,
    const int root,
    MPI_Comm comm)
{
    assert(A->NB > 0);
    assert(A->MB > 0);

    int *indexb = bml_allocate_memory(sizeof(int) * A->NB * A->MB);
    if (bml_getMyRank() == root)
        memcpy(indexb, A->indexb, A->NB * A->MB * sizeof(int));

    int *nnzb = bml_allocate_memory(sizeof(int) * A->NB);
    if (bml_getMyRank() == root)
        memcpy(nnzb, A->nnzb, sizeof(int) * A->NB);

    MPI_Bcast(nnzb, A->NB, MPI_INT, root, comm);
    MPI_Bcast(indexb, A->NB * A->MB, MPI_INT, root, comm);

    // bcast matrix elements
    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;
    REAL_T *values = bml_allocate_memory(sizeof(REAL_T) * A->N * A->M);
    if (bml_getMyRank() == root)
    {
        REAL_T *pvalues = values;
        for (int ib = 0; ib < A->NB; ib++)
        {
            for (int jp = 0; jp < A->nnzb[ib]; jp++)
            {
                int ind = ROWMAJOR(ib, jp, A->NB, A->MB);
                int jb = A->indexb[ind];
                int nelements = A->bsize[ib] * A->bsize[jb];
                memcpy(pvalues, A_ptr_value[ind], nelements * sizeof(REAL_T));
                pvalues += nelements;
            }
        }
    }
    MPI_Bcast(values, A->N * A->M, MPI_T, root, comm);

    bml_clear_ellblock(A);

    // copy data into bml matrix
    memcpy(A->nnzb, nnzb, sizeof(int) * A->NB);
    memcpy(A->indexb, indexb, A->NB * A->MB * sizeof(int));

    REAL_T *pvalues = values;

    for (int ib = 0; ib < A->NB; ib++)
    {
        for (int jp = 0; jp < A->nnzb[ib]; jp++)
        {
            int ind = ROWMAJOR(ib, jp, A->NB, A->MB);
            int jb = A->indexb[ind];
            int nelements = A->bsize[ib] * A->bsize[jb];
            A_ptr_value[ind] =
                TYPED_FUNC(bml_allocate_block_ellblock) (A, ib, nelements);
            memcpy(A_ptr_value[ind], pvalues, nelements * sizeof(REAL_T));
            pvalues += nelements;
        }
    }

    bml_free_memory(nnzb);
    bml_free_memory(indexb);
    bml_free_memory(values);
}
#endif
