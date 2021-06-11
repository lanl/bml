#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_submatrix.h"
#include "../bml_types.h"
#include "bml_allocate_csr.h"
#include "bml_submatrix_csr.h"
#include "bml_types_csr.h"
#include "bml_getters_csr.h"
#include "bml_setters_csr.h"

#include <complex.h>
#include <stdio.h>
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
bml_matrix_csr_t
    * TYPED_FUNC(bml_extract_submatrix_csr) (bml_matrix_csr_t * A,
                                             int irow, int icol,
                                             int B_N, int B_M)
{
    bml_matrix_csr_t *B;
    B = TYPED_FUNC(bml_zero_matrix_csr) (B_N, B_M, A->distribution_mode);

    // loop over subset of rows of A
    for (int i = irow; i < irow + B_N; i++)
    {
        int nz = TYPED_FUNC(csr_get_nnz) (A->data_[i]);
        int *cols = TYPED_FUNC(csr_get_column_indexes) (A->data_[i]);
        REAL_T *vals = TYPED_FUNC(csr_get_column_entries) (A->data_[i]);
        // extract data between icol and icol+B_N
        int *newcols = bml_noinit_allocate_memory(B_N * sizeof(int));
        REAL_T *newvals = bml_noinit_allocate_memory(B_N * sizeof(REAL_T));
        int count = 0;
        for (int j = 0; j < nz; j++)
        {
            if (cols[j] >= icol && cols[j] < icol + B_N)
            {
                newcols[count] = cols[j] - icol;
                newvals[count] = vals[j];
                count++;
            }
        }

        TYPED_FUNC(bml_set_sparse_row_csr) (B, i - irow, count, newcols,
                                            newvals, 0.);
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
    bml_assign_submatrix_csr) (
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    int irow,
    int icol)
{
    int B_N = B->N_;

    // loop over rows of B
    for (int i = 0; i < B_N; i++)
    {
        int nz = TYPED_FUNC(csr_get_nnz) (B->data_[i]);
        int *cols = TYPED_FUNC(csr_get_column_indexes) (B->data_[i]);
        for (int j = 0; j < nz; j++)
        {
            assert(cols[j] < B_N);
        }
        REAL_T *vals = TYPED_FUNC(csr_get_column_entries) (B->data_[i]);

        for (int j = 0; j < nz; j++)
        {
            TYPED_FUNC(csr_set_row_element_new) (A->data_[i + irow],
                                                 cols[j] + icol, &vals[j]);
        }
    }
}
