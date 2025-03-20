#ifdef BML_USE_MAGMA
#include <stdbool.h>
#include "magma_v2.h"
#endif

#include "../../internal-blas/bml_gemm.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_multiply.h"
#include "../bml_parallel.h"
#include "../bml_trace.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_multiply_dense.h"
#include "bml_trace_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MKL_GPU
#include "stdio.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

/** Matrix multiply.
 *
 * \f$ C \leftarrow \alpha \, A \, B + \beta C \f$
 *
 * \ingroup multiply_group
 *
 * Note that the internal storage order of the dense matrix is
 * row-major while the standard implementation of gemm assumes column
 * major order. We therefore use a storage order transpose to the gemm
 * implementation. Instead of multiplying \f$ A \, B f$ we will
 * instead calculate \f$ B \, A \f$ and get:
 *
 * \f$ C^{T} \leftarrow \alpha \, B^{T} \, A^{T} + \beta C^{T} \f$
 *
 * \f$ C \f$ will then become:
 *
 * \f$ C \leftarrow \alpha \, \left( B^{T} \, A^{T} \right)^{T} + \beta C \f$
 * \f$ C \leftarrow \alpha \, A \, B + \beta C \f$
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param alpha Scalar factor multiplied by A * B
 * \param beta Scalar factor multiplied by C
 */
void TYPED_FUNC(
    bml_multiply_dense) (
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C,
    double alpha,
    double beta)
{
#ifdef BML_USE_MAGMA
    MAGMA_T alpha_ = MAGMACOMPLEX(MAKE) (alpha, 0.);
    MAGMA_T beta_ = MAGMACOMPLEX(MAKE) (beta, 0.);

    MAGMA(gemm) (MagmaNoTrans, MagmaNoTrans,
                 A->N, A->N, A->N, alpha_, B->matrix, B->ld,
                 A->matrix, A->ld, beta_, C->matrix, C->ld, bml_queue());
    magma_queue_sync(bml_queue());
#elif defined(MKL_GPU)
    int sizea = A->N * A->N;
    int dnum = 0;

    REAL_T *A_matrix = (REAL_T *) A->matrix;
    REAL_T *B_matrix = (REAL_T *) B->matrix;
    REAL_T *C_matrix = (REAL_T *) C->matrix;

    MKL_T alpha_;
    MKL_T beta_;

    MKL_REAL(alpha_) = (REAL_T) alpha;
    MKL_REAL(beta_) = (REAL_T) beta;
    MKL_IMAG(alpha_);
    MKL_IMAG(beta_);

    {
        // run gemm on gpu, use standard oneMKL interface within a variant dispatch construct
#pragma omp target variant dispatch device(dnum) use_device_ptr(B_matrix, A_matrix, C_matrix)
        {
            G_BLAS(gemm) (CblasColMajor, CblasNoTrans, CblasNoTrans, A->N,
                          A->N, A->N, MKL_ADDRESS(alpha_), B_matrix, A->N,
                          A_matrix, A->N, MKL_ADDRESS(beta_), C_matrix, A->N);
        }
    }
#else
    REAL_T alpha_ = (REAL_T) alpha;
    REAL_T beta_ = (REAL_T) beta;

    TYPED_FUNC(bml_gemm) ("N", "N", &A->N, &A->N, &A->N, &alpha_, B->matrix,
                          &A->N, A->matrix, &A->N, &beta_, C->matrix, &A->N);
#endif

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
void *TYPED_FUNC(
    bml_multiply_x2_dense) (
    bml_matrix_dense_t * X,
    bml_matrix_dense_t * X2)
{
    double *trace = bml_allocate_memory(sizeof(double) * 2);

    trace[0] = TYPED_FUNC(bml_trace_dense) (X);
    TYPED_FUNC(bml_multiply_dense) (X, X, X2, 1.0, 0.0);
    trace[1] = TYPED_FUNC(bml_trace_dense) (X2);

    return trace;
}

/** Matrix multiply.
 *
 * \f$ C \leftarrow A \, B \f$
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 */
void TYPED_FUNC(
    bml_multiply_AB_dense) (
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C)
{
    TYPED_FUNC(bml_multiply_dense) (A, B, C, 1.0, 0.0);
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
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B,
    bml_matrix_dense_t * C)
{
    REAL_T alpha = (REAL_T) 1.0;
    REAL_T beta = (REAL_T) 0.0;

    TYPED_FUNC(bml_gemm) ("T", "T", &A->N, &A->N, &A->N, &alpha, A->matrix,
                          &A->N, B->matrix, &A->N, &beta, C->matrix, &A->N);
}
