#include "../blas.h"
#include "../macros.h"
#include "../typed.h"
#include "bml_add.h"
#include "bml_add_dense.h"
#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

/** Matrix addition.
 *
 * \f$ A = \alpha A + \beta B \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param alpha Scalar factor multiplied by A
 * \param beta Scalar factor multiplied by B
 */
void TYPED_FUNC(
    bml_add_dense) (
    bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta)
{
    REAL_T alpha_ = alpha;
    REAL_T beta_ = beta;
    int nElems = B->N * B->N;
    int inc = 1;

    C_BLAS(SCAL) (&nElems, &alpha_, A->matrix, &inc);
    C_BLAS(AXPY) (&nElems, &beta_, B->matrix, &inc, A->matrix, &inc);
}

/** Matrix addition and calculate TrNorm.
 *
 * \f$ A = \alpha A + \beta B \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param alpha Scalar factor multiplied by A
 * \param beta Scalar factor multiplied by B
 */
double TYPED_FUNC(
    bml_add_norm_dense) (
    bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta)
{
    double trnorm = 0.0;
    REAL_T *B_matrix = (REAL_T *) B->matrix;

    for (int i = 0; i < A->N * A->N; i++)
    {
        trnorm += B_matrix[i] * B_matrix[i];
    }

    TYPED_FUNC(bml_add_dense) (A, B, alpha, beta);

    return trnorm;
}

/** Matrix addition.
 *
 * \f$ A = A + \beta \mathrm{Id} \f$
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param beta Scalar factor multiplied by A
 */
void TYPED_FUNC(
    bml_add_identity_dense) (
    bml_matrix_dense_t * A,
    const double beta)
{
    REAL_T beta_ = beta;
    REAL_T *A_matrix = (REAL_T *) A->matrix;
    for (int i = 0; i < A->N; i++)
    {
        A_matrix[ROWMAJOR(i, i, A->N, A->N)] += beta_;
    }
}
