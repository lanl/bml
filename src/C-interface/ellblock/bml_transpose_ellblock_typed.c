#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_parallel.h"
#include "../bml_transpose.h"
#include "../bml_types.h"
#include "bml_allocate_ellblock.h"
#include "bml_transpose_ellblock.h"
#include "bml_types_ellblock.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Transpose a matrix.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposed
 *  \return the transposed A
 */
bml_matrix_ellblock_t
    * TYPED_FUNC(bml_transpose_new_ellblock) (bml_matrix_ellblock_t * A)
{
    int NB = A->NB;
    int MB = A->MB;
    int *bsize = A->bsize;
    bml_distribution_mode_t distmode = A->distribution_mode;

    bml_matrix_ellblock_t *B =
        TYPED_FUNC(bml_block_matrix_ellblock) (NB, MB, A->M,
                                               bsize,
                                               distmode);

    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;
    int *A_indexb = A->indexb;
    int *A_nnzb = A->nnzb;

    REAL_T **B_ptr_value = (REAL_T **) B->ptr_value;
    int *B_indexb = B->indexb;
    int *B_nnzb = B->nnzb;

    for (int ib = 0; ib < NB; ib++)
    {
        for (int jp = 0; jp < A_nnzb[ib]; jp++)
        {
            int indA = ROWMAJOR(ib, jp, NB, MB);
            int jb = A_indexb[indA];
            int indB = ROWMAJOR(jb, B_nnzb[jb], NB, MB);
            //add a new block in B
            B_indexb[indB] = ib;
            int nelements = bsize[ib] * bsize[jb];
            B_ptr_value[indB] =
                TYPED_FUNC(bml_allocate_block_ellblock) (B, ib, nelements);
            B_nnzb[jb]++;
            //transpose block
            REAL_T *B_value = B_ptr_value[indB];
            REAL_T *A_value = A_ptr_value[indA];
            for (int ii = 0; ii < bsize[ib]; ii++)
                for (int jj = 0; jj < bsize[jb]; jj++)
                    B_value[ROWMAJOR(jj, ii, bsize[jb], bsize[ib])] =
                        A_value[ROWMAJOR(ii, jj, bsize[ib], bsize[jb])];
        }
    }

    return B;
}

/** swap block row entries in position iposb and jposb.
 *
 * column indexes and non-zero entries are swapped
 *
 * \ingroup transpose_group
 *
 * \param A The matrix.
 */
void TYPED_FUNC(
    ellblock_swap_block_row_entries) (
    bml_matrix_ellblock_t * A,
    const int block_row,
    const int iposb,
    const int jposb)
{
    if (iposb == jposb)
        return;

    int NB = A->NB;
    int MB = A->MB;
    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;
    int *A_indexb = A->indexb;

    REAL_T *tmp = A_ptr_value[ROWMAJOR(block_row, iposb, NB, MB)];  
    int itmp = A_indexb[ROWMAJOR(block_row, iposb, NB, MB)];  

    /* swap block data pointers */
    A_ptr_value[ROWMAJOR(block_row, iposb, NB, MB)] = A_ptr_value[ROWMAJOR(block_row, jposb, NB, MB)];
    A_ptr_value[ROWMAJOR(block_row, jposb, NB, MB)] = tmp;
    A_indexb[ROWMAJOR(block_row, iposb, NB, MB)] = A_indexb[ROWMAJOR(block_row, jposb, NB, MB)];
    A_indexb[ROWMAJOR(block_row, jposb, NB, MB)] = itmp;
}

/** Transpose a matrix in place.
 *
 *  \ingroup transpose_group
 *
 *  \param A The matrix to be transposeed
 *  \return the transposed A
 */
void TYPED_FUNC(
    bml_transpose_ellblock) (
    bml_matrix_ellblock_t * A)
{
    int NB = A->NB;
    int MB = A->MB;

    REAL_T **A_ptr_value = (REAL_T **) A->ptr_value;
    int *A_indexb = A->indexb;
    int *A_nnzb = A->nnzb;
    int *bsize = A->bsize;

    int max_block_size = 0;
    for(int i=0; i<NB; i++)
    {
        max_block_size = max_block_size >= bsize[i] ? max_block_size : bsize[i];
    }
    const int BUFFER_SIZE = max_block_size * max_block_size;

    REAL_T buffer[BUFFER_SIZE];

    int nz_t[NB];
    memset(nz_t, 0, sizeof(int) * NB);

    for (int ib = 0; ib < NB; ib++)
    {
        const int innzb = A_nnzb[ib];
        int iposb = innzb - 1;
        while (iposb >= nz_t[ib])
        {
            const int jb = A_indexb[ROWMAJOR(ib, iposb, NB, MB)];
            if (jb > ib)
            {
                const int jnnzb = A_nnzb[jb];
                const int jbstart = nz_t[jb];
                int found = 0;
                /* search for symmetric position */
                for (int jposb = jbstart; jposb < jnnzb; jposb++)
                {
                    /* symmetric position found so just swap block entries */
                    if (A_indexb[ROWMAJOR(jb, jposb, NB, MB)] == ib)
                    {
                        /* swap and transpose block row entries in row ib position iposb 
                         * row jb position jposb. 
                        */
                        REAL_T *A_value_ib = A_ptr_value[ROWMAJOR(ib, iposb, NB, MB)];
                        REAL_T *A_value_jb = A_ptr_value[ROWMAJOR(jb, jposb, NB, MB)];

                        for (int ii = 0; ii < bsize[ib]; ii++)
                        {
                            for (int jj = 0; jj < bsize[jb]; jj++)
                            {
                                REAL_T tmp = A_value_ib[ROWMAJOR(ii, jj, bsize[ib], bsize[jb])];
                                A_value_ib[ROWMAJOR(ii, jj, bsize[ib], bsize[jb])] =
                                A_value_jb[ROWMAJOR(jj, ii, bsize[jb], bsize[ib])];
                                A_value_jb[ROWMAJOR(jj, ii, bsize[jb], bsize[ib])] = tmp;
                            }
                        }

                        /* swap position in row i to process next entry */
                        TYPED_FUNC(ellblock_swap_block_row_entries) (A, ib, iposb, nz_t[ib]);
                        /* swap position in row j */
                        TYPED_FUNC(ellblock_swap_block_row_entries) (A, jb, jposb,
                                                          nz_t[jb]);
                        /* update nonzero count */
                        nz_t[ib]++;
                        nz_t[jb]++;
                        found = 1;
                        break;
                    }
                }
                if (!found)
                {
                    /* nonsymmetric entry. Insert block entries and swap position */
                    const int nnz_jb = A_nnzb[jb];
                    A_ptr_value[ROWMAJOR(jb, A_nnzb[jb], NB, MB)] = A_ptr_value[ROWMAJOR(ib, iposb, NB, MB)];
                    A_indexb[ROWMAJOR(jb, A_nnzb[jb], NB, MB)] = ib;
                    A_nnzb[jb]++;
                    
                    /* transpose inserted block */
                    REAL_T *A_value_jb = A_ptr_value[ROWMAJOR(jb, nnz_jb, NB, MB)];
                    for (int ii = 0; ii < bsize[ib]; ii++)
                    {
                        for (int jj = 0; jj < bsize[jb]; jj++)
                        {
                            buffer[COLMAJOR(ii, jj, bsize[ib], bsize[jb])] = 
                            A_value_jb[ROWMAJOR(ii, jj, bsize[ib], bsize[jb])];
                        }
                    }
                    const int blocksize = bsize[ib] * bsize[jb];
                    memcpy(A_value_jb, buffer, blocksize * sizeof(REAL_T));

                    /* swap position in updated row j */
                    const int nzposb = A_nnzb[jb] - 1;
                    TYPED_FUNC(ellblock_swap_block_row_entries) (A, jb, nzposb,
                                                      nz_t[jb]);
                    /* update nonzero count */
                    nz_t[jb]++;
                    A_nnzb[ib]--;
                    /* update ipos */
                    iposb--;
                }
            }
            else if (jb < ib)
            {
                // insert block entries in block row jb
                const int nnz_jb = A_nnzb[jb];
                A_ptr_value[ROWMAJOR(jb, nnz_jb, NB, MB)] = A_ptr_value[ROWMAJOR(ib, iposb, NB, MB)];
                A_indexb[ROWMAJOR(jb, nnz_jb, NB, MB)] = ib;
                A_nnzb[jb]++;

                /* transpose inserted block */
                REAL_T *A_value_jb = A_ptr_value[ROWMAJOR(jb, nnz_jb, NB, MB)];
                for (int ii = 0; ii < bsize[ib]; ii++)
                {
                    for (int jj = 0; jj < bsize[jb]; jj++)
                    {
                        buffer[COLMAJOR(ii, jj, bsize[ib], bsize[jb])] = 
                        A_value_jb[ROWMAJOR(ii, jj, bsize[ib], bsize[jb])];
                    }
                }
                const int blocksize = bsize[ib] * bsize[jb];
                memcpy(A_value_jb, buffer, blocksize * sizeof(REAL_T)); 

                /* swap position in updated row j */
                const int nzposb = A_nnzb[jb] - 1;
                TYPED_FUNC(ellblock_swap_block_row_entries) (A, jb, nzposb,
                                                  nz_t[jb]);
                /* update nonzero count */
                nz_t[jb]++;
                A_nnzb[ib]--;
                /* update ipos */
                iposb--;
            }
            else /* jb == ib*/
            {
                REAL_T *A_value_ib = A_ptr_value[ROWMAJOR(ib, iposb, NB, MB)];
                for (int ii = 0; ii < bsize[ib]; ii++)
                {
                    for (int jj = ii; jj < bsize[ib]; jj++)
                    {
                        REAL_T tmp = A_value_ib[ROWMAJOR(ii, jj, bsize[ib], bsize[ib])];
                        A_value_ib[ROWMAJOR(ii, jj, bsize[ib], bsize[ib])] =
                        A_value_ib[ROWMAJOR(jj, ii, bsize[ib], bsize[ib])];
                        A_value_ib[ROWMAJOR(jj, ii, bsize[ib], bsize[ib])] = tmp;
                    }
                }
                /* swap position in row ib */
                TYPED_FUNC(ellblock_swap_block_row_entries) (A, ib, iposb,
                                                  nz_t[ib]);
                /* update nonzero count */
                nz_t[ib]++;
            }
        }
    }
}