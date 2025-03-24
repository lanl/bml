#include "../../macros.h"
#include "../../typed.h"
#include "bml_allocate_dense.h"
#include "bml_types_dense.h"

#ifdef BML_USE_MAGMA
#include <stdbool.h> //define boolean data type for magma 
#include "magma_v2.h"
#endif

#include <memory.h>
#include <complex.h>

/** Extract submatrix into new matrix of same format
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A to extract submatrix from
 * \param irow Index of first row to extract
 * \param icol Index of first column to extract
 * \param B_N Number of rows/columns to extract
 * \param B_M unused
 */
bml_matrix_dense_t
    * TYPED_FUNC(bml_extract_submatrix_dense) (bml_matrix_dense_t * A,
                                               int irow, int icol,
                                               int B_N, int B_M)
{
    bml_matrix_dimension_t matrix_dimension = { B_N, B_N, B_N };
    bml_matrix_dense_t *B =
        TYPED_FUNC(bml_zero_matrix_dense) (matrix_dimension,
                                           A->distribution_mode);

    int A_N = A->N;
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;

#ifdef BML_USE_MAGMA
    MAGMA(copymatrix) (B_N, B_N, (MAGMA_T *) (A_matrix + irow * A->ld + icol),
                       A->ld, (MAGMA_T *) B_matrix, B->ld, bml_queue());
#else
    // loop over subset of rows of A
    for (int i = irow; i < irow + B_N; i++)
    {
        for (int j = icol; j < icol + B_N; j++)
        {
            B_matrix[ROWMAJOR(i - irow, j - icol, B_N, B_N)] =
                A_matrix[ROWMAJOR(i, j, A_N, A_N)];
        }
    }
#endif
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
    bml_assign_submatrix_dense) (
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    int irow,
    int icol)
{
    int A_N = A->N;
    int B_N = B->N;
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;

#ifdef BML_USE_MAGMA
    MAGMA(copymatrix) (B_N, B_N, (MAGMA_T *) B_matrix, B->ld,
                       (MAGMA_T *) (A_matrix + irow * A->ld + icol), A->ld,
                       bml_queue());
#else
    // loop over rows of B
    for (int i = 0; i < B_N; i++)
    {
        int offsetA = ROWMAJOR(i + irow, icol, A_N, A_N);
        int offsetB = ROWMAJOR(i, 0, B_N, B_N);
        memcpy(A_matrix + offsetA, B_matrix + offsetB, B_N * sizeof(REAL_T));
    }
#endif
}
