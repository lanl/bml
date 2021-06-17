#include "bml_utilities_distributed2d.h"
#include "bml_allocate_distributed2d.h"
#include "../bml_introspection.h"
#include "../bml_utilities.h"
#include "../bml_allocate.h"
#include "../bml_parallel.h"
#include "../bml_submatrix.h"
#include "../bml_copy.h"
#include "../ellblock/bml_types_ellblock.h"
#include "../ellblock/bml_allocate_ellblock.h"
#include "../bml_logger.h"

#include <string.h>
#include <assert.h>

void
bml_read_bml_matrix_distributed2d(
    bml_matrix_distributed2d_t * A,
    char *filename)
{
    // create a big matrix that can store all the blocks
    int B_NB;
    int B_MB;
    int lnb;
    int *bsizes;
    bml_matrix_t *Alocal = bml_get_local_matrix(A);
    bml_matrix_t *B;
    switch (bml_get_type(A->matrix))
    {
            // special case for ellblock: we need block sizes to exactly
            // match block sizes of local matrices
        case ellblock:
            lnb = ((bml_matrix_ellblock_t *) Alocal)->NB;
            B_NB = lnb * A->npcols;
            B_MB = ((bml_matrix_ellblock_t *) Alocal)->MB * A->npcols;
            bsizes = bml_noinit_allocate_memory(B_NB * sizeof(int));
            for (int p = 0; p < A->npcols; p++)
                memcpy(bsizes + p * lnb,
                       ((bml_matrix_ellblock_t *) Alocal)->bsize,
                       lnb * sizeof(int));
            B = bml_block_matrix_ellblock(A->matrix_precision, B_NB, B_MB,
                                          A->M, bsizes, sequential);
            break;
        default:
            B = bml_zero_matrix(bml_get_type(A->matrix),
                                A->matrix_precision, A->N, A->M, sequential);
            break;
    }

    // read data into "big" matrix B by one task only
    if (A->mpitask == 0)
        bml_read_bml_matrix(B, filename);

    bml_mpi_bcast_matrix(B, 0, A->comm);

    // extract local submatrix out of replicated "big" B matrix
    int irow = A->myprow * A->n;
    int icol = A->mypcol * A->n;
    bml_matrix_t *C =
        bml_extract_submatrix(B, irow, icol, A->n, A->M / A->npcols);

    bml_copy(C, A->matrix);

    bml_deallocate(&B);
    bml_deallocate(&C);
}

void
bml_write_bml_matrix_distributed2d(
    bml_matrix_distributed2d_t * A,
    char *filename)
{
    int B_NB;
    int B_MB;
    int lnb;
    int *bsizes;
    bml_matrix_t *Alocal = bml_get_local_matrix(A);

    // task 0 collects all blocks and write matrix
    if (A->mpitask == 0)
    {
        bml_matrix_t *B;

        // create a big matrix that can store all the blocks
        switch (bml_get_type(A->matrix))
        {
                // special case for ellblock: we need block sizes to exactly
                // match block sizes of local matrices
            case ellblock:
                lnb = ((bml_matrix_ellblock_t *) Alocal)->NB;
                B_NB = lnb * A->npcols;
                B_MB = ((bml_matrix_ellblock_t *) Alocal)->MB * A->npcols;
                bsizes = bml_noinit_allocate_memory(B_NB * sizeof(int));
                // block sizes for "big" matrix made of block sizes
                // of local matrices
                for (int p = 0; p < A->npcols; p++)
                    memcpy(bsizes + p * lnb,
                           ((bml_matrix_ellblock_t *) Alocal)->bsize,
                           lnb * sizeof(int));
                B = bml_block_matrix_ellblock(A->matrix_precision, B_NB, B_MB,
                                              A->M, bsizes, sequential);
                break;
            default:
                B = bml_noinit_matrix(bml_get_type(A->matrix),
                                      A->matrix_precision, A->N, A->M,
                                      sequential);
                break;
        }

        // assign "local" matrices into "big" matrixe
        bml_assign_submatrix(B, A->matrix, 0, 0);
        int C_N = A->n;
        int C_M = A->M / A->npcols;
        for (int itask = 1; itask < A->ntasks; itask++)
        {
            bml_matrix_t *C = bml_noinit_matrix(bml_get_type(A->matrix),
                                                A->matrix_precision, C_N,
                                                C_M,
                                                sequential);
            bml_mpi_recv(C, itask, A->comm);
            int irow = (itask / A->npcols) * C_N;
            int icol = (itask % A->npcols) * C_N;
            bml_assign_submatrix(B, C, irow, icol);
            bml_deallocate(&C);
        }

        // print "big" matrix containing all the distributed submatrices
        bml_write_bml_matrix(B, filename);
        bml_deallocate(&B);
    }
    else
    {
        // send local submatrix to task 0
        bml_mpi_send(A->matrix, 0, A->comm);
    }
}
