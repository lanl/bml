#include "../typed.h"
#include "../blas.h"
#include "bml_multiply.h"
#include "bml_types.h"
#include "bml_multiply_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Matrix multiply.
 *
 * C = alpha * A * B + beta * C
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 *  \param alpha Scalar factor multiplied by A * B
 *  \param beta Scalar factor multiplied by C
 */
void TYPED_FUNC(
    bml_multiply_dense) (
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    bml_matrix_dense_t * C,
    const double alpha,
    const double beta)
{
    REAL_T alpha_ = (REAL_T) alpha;
    REAL_T beta_ = (REAL_T) beta;
    C_BLAS(GEMM) ("N", "N", &A->N, &A->N, &A->N, &alpha_, A->matrix,
                  &A->N, B->matrix, &A->N, &beta_, C->matrix, &A->N);
}

/** Matrix multiply.
 *
 * X2 = X * X
 *
 *  \ingroup multiply_group
 *
 *  \param X Matrix X
 *  \param X2 MatrixX2
 */
void TYPED_FUNC(
    bml_multiply_x2_dense) (
    const bml_matrix_dense_t * X,
    bml_matrix_dense_t * X2)
{
    REAL_T alpha = (REAL_T) 1.0;
    REAL_T beta = (REAL_T) 1.0;
    C_BLAS(GEMM) ("N", "N", &X->N, &X->N, &X->N, &alpha, X->matrix,
                  &X->N, X->matrix, &X->N, &beta, X2->matrix, &X->N);
}

/** Matrix multiply.
 *
 * C = A * B
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 */
void TYPED_FUNC(
    bml_multiply_AB_dense) (
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    bml_matrix_dense_t * C)
{
    REAL_T alpha = (REAL_T) 1.0;
    REAL_T beta = (REAL_T) 0.0;
    C_BLAS(GEMM) ("N", "N", &A->N, &A->N, &A->N, &alpha, A->matrix,
                  &A->N, B->matrix, &A->N, &beta, C->matrix, &A->N);
}

/** Matrix multiply.
 *
 * This routine is provided for completeness.
 *
 * C = A * B
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 */
void TYPED_FUNC(
    bml_multiply_adjust_AB_dense) (
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    bml_matrix_dense_t * C)
{
    REAL_T alpha = (REAL_T) 1.0;
    REAL_T beta = (REAL_T) 0.0;
    C_BLAS(GEMM) ("N", "N", &A->N, &A->N, &A->N, &alpha, A->matrix,
                  &A->N, B->matrix, &A->N, &beta, C->matrix, &A->N);
}

