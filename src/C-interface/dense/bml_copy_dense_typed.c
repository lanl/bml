#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_copy.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_types_dense.h"

#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#endif

#ifdef MKL_GPU
#include "stdio.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/** Copy a dense matrix - result in new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_dense_t *TYPED_FUNC(
    bml_copy_dense_new) (
    bml_matrix_dense_t * A)
{
    bml_matrix_dimension_t matrix_dimension = { A->N, A->N, A->N };
    bml_matrix_dense_t *B =
        TYPED_FUNC(bml_zero_matrix_dense) (matrix_dimension,
                                           A->distribution_mode);
#ifdef BML_USE_MAGMA
    MAGMA(copymatrix) (A->N, A->N, A->matrix, A->ld,
                       B->matrix, B->ld, bml_queue());
#else
#ifdef MKL_GPU
// pull from GPU
    int N = A->N;
    REAL_T *A_matrix = A->matrix;
#pragma omp target update from(A_matrix[0:N*N])
#endif
    memcpy(B->matrix, A->matrix, sizeof(REAL_T) * A->N * A->N);
#ifdef MKL_GPU
    int sizea = B->N * B->N;
    int dnum = 0;

    REAL_T *B_matrix = (REAL_T *) B->matrix;
    // allocate and offload the matrix to GPU
#pragma omp target enter data map(alloc:B_matrix[0:N*N]) device(dnum)
#pragma omp target update to(B_matrix[0:N*N])
#endif // end of MKL_GPU
#endif
    bml_copy_domain(A->domain, B->domain);
    bml_copy_domain(A->domain2, B->domain2);
    return B;
}

/** Copy a dense matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A
 */
void TYPED_FUNC(
    bml_copy_dense) (
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B)
{
#ifdef BML_USE_MAGMA
    MAGMA(copymatrix) (A->N, A->N, A->matrix, A->ld,
                       B->matrix, B->ld, bml_queue());
#else
#ifdef MKL_GPU
// pull from GPU
    int N = A->N;
    REAL_T *A_matrix = A->matrix;
#pragma omp target update from(A_matrix[0:N*N])
#endif
    memcpy(B->matrix, A->matrix, sizeof(REAL_T) * A->N * A->N);
#endif
#ifdef MKL_GPU
    int sizea = B->N * B->N;
    int dnum = 0;

    REAL_T *B_matrix = (REAL_T *) B->matrix;
    // allocate and offload the matrix to GPU
#pragma omp target enter data map(alloc:B_matrix[0:N*N])
#pragma omp target update to(B_matrix[0:N*N])
#endif // end of MKL_GPU
    if (A->distribution_mode == B->distribution_mode)
    {
        bml_copy_domain(A->domain, B->domain);
        bml_copy_domain(A->domain2, B->domain2);
    }
}

/** Reorder a dense matrix using a permutation vector.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be reordered
 *  \param perm The permutation vector
 */
void TYPED_FUNC(
    bml_reorder_dense) (
    bml_matrix_dense_t * A,
    int *perm)
{
    int N = A->N;

    bml_matrix_dense_t *B = TYPED_FUNC(bml_copy_dense_new) (A);
    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;

#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
    // Reorder rows - need to copy
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        memcpy(&A_matrix[ROWMAJOR(perm[i], 0, N, N)],
               &B_matrix[ROWMAJOR(i, 0, N, N)], N * sizeof(REAL_T));
    }

    // Reorder elements in each row - just change index
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A_matrix[ROWMAJOR(i, j, N, N)] =
                B_matrix[ROWMAJOR(i, perm[j], N, N)];
        }
    }
#ifdef MKL_GPU
    int dnum = 0;

    // allocate and offload the matrix to GPU
//#pragma omp target enter data map(alloc:A_matrix[0:N*N]) device(dnum)
#pragma omp target update to(A_matrix[0:N*N])
#endif // end of MKL_GPU

    bml_deallocate_dense(B);
}
