#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_submatrix.h"
#include "../bml_types.h"
#include "../dense/bml_allocate_dense.h"
#include "bml_allocate_ellblock.h"
#include "bml_submatrix_ellblock.h"
#include "bml_types_ellblock.h"

#include <stdlib.h>
#include <assert.h>

/** Extract submatrix into new matrix of same format
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A to extract submatrix from
 * \param irow Index of first row to extract
 * \param icol Index of first column to extract
 * \param B_N Number of rows/columns to extract
 * \param B_M Max number of non-zero elemnts/row in exttacted matrix
 */
bml_matrix_ellblock_t
    * TYPED_FUNC(bml_extract_submatrix_ellblock) (bml_matrix_ellblock_t * A,
                                                  int irow, int icol,
                                                  int B_N, int B_M)
{
    int A_NB = A->NB;
    int A_MB = A->MB;

    int *A_indexb = A->indexb;
    int *A_nnzb = A->nnzb;
    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;

    bml_matrix_ellblock_t *B;
    B = TYPED_FUNC(bml_zero_matrix_ellblock) (B_N, B_M, A->distribution_mode);

    int *B_indexb = B->indexb;
    int *B_nnzb = B->nnzb;
    REAL_T **B_ptr_value = (REAL_T **) B->ptr_value;

    int B_NB = B->NB;
    int B_MB = B->MB;

    int *bsize = A->bsize;
    int count = 0;
    int irowb = 0;
    while (count < irow)
    {
        count += bsize[irowb];
        irowb++;
    }
    assert(count == irow);

    count = 0;
    int icolb = 0;
    while (count < icol)
    {
        count += bsize[icolb];
        icolb++;
    }
    assert(count == icol);

    //count number of block rows/cols in B
    count = 0;
    int nb = 0;
    while (count < B_N)
    {
        count += bsize[nb];
        nb++;
    }

    // loop over subset of block rows of A
    for (int ib = irowb; ib < irowb + nb; ib++)
    {
        for (int jp = 0; jp < A_nnzb[ib]; jp++)
        {
            int jb = A_indexb[ROWMAJOR(ib, jp, A_NB, A_MB)];
            // check if block overlaps with B
            if (jb >= icolb && jb < icolb + nb)
            {
                int nelements = bsize[ib] * bsize[jb];
                int indB =
                    ROWMAJOR(ib - irowb, B_nnzb[ib - irowb], B_NB, B_MB);
                B_indexb[indB] = jb - icolb;
                B_ptr_value[indB] =
                    TYPED_FUNC(bml_allocate_block_ellblock) (B, ib - irowb,
                                                             nelements);
                REAL_T *B_value = B_ptr_value[indB];
                int indA = ROWMAJOR(ib, jp, A_NB, A_MB);
                REAL_T *A_value = A_ptr_value[indA];
                for (int kk = 0; kk < nelements; kk++)
                {
                    B_value[kk] = A_value[kk];
                }

                B_nnzb[ib - irowb]++;
            }
        }
    }

    return B;
}

/** Assign a block B into matrix A
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param irow First row where to insert block B
 * \param icol Offset column to insert block B
 */
void TYPED_FUNC(
    bml_assign_submatrix_ellblock) (
    bml_matrix_ellblock_t * A,
    bml_matrix_ellblock_t * B,
    int irow,
    int icol)
{
    int A_NB = A->NB;
    int A_MB = A->MB;
    int *A_indexb = A->indexb;
    int *A_nnzb = A->nnzb;
    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;

    int B_NB = B->NB;
    int B_MB = B->MB;
    int *B_indexb = B->indexb;
    int *B_nnzb = B->nnzb;
    REAL_T **B_ptr_value = (REAL_T **) B->ptr_value;

    int *bsize = A->bsize;
    int count = 0;
    int irowb = 0;
    while (count < irow)
    {
        count += bsize[irowb];
        irowb++;
    }
    assert(count == irow);

    count = 0;
    int icolb = 0;
    while (count < icol)
    {
        count += bsize[icolb];
        icolb++;
    }
    assert(count == icol);

    // loop over block rows of B
    for (int ib = 0; ib < B_NB; ib++)
    {
        for (int jp = 0; jp < B_nnzb[ib]; jp++)
        {
            int jb = B_indexb[ROWMAJOR(ib, jp, B_NB, B_MB)];
            int nelements = bsize[ib] * bsize[jb];
            int indA = ROWMAJOR(ib + irowb, A_nnzb[ib + irowb], A_NB, A_MB);
            int indB = ROWMAJOR(ib, jp, B_NB, B_MB);
            A_ptr_value[indA] =
                TYPED_FUNC(bml_allocate_block_ellblock) (A, ib + irowb,
                                                         nelements);
            A_indexb[indA] = jb + icolb;

            REAL_T *B_value = B_ptr_value[indB];
            REAL_T *A_value = A_ptr_value[indA];
            for (int kk = 0; kk < nelements; kk++)
            {
                A_value[kk] = B_value[kk];
            }

            A_nnzb[ib + irowb]++;
        }
    }
}
