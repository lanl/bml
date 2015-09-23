#include "../blas.h"
#include "../bml_allocate.h"
#include "../bml_add.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_add_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Matrix addition.
 *
 * A = alpha * A + beta * B
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param alpha Scalar factor multiplied by A
 *  \param beta Scalar factor multiplied by B
 */
void bml_add_dense(const bml_matrix_dense_t *A, const bml_matrix_dense_t *B, const double alpha, const double beta)
{
    float alpha_s, beta_s;

    int nElems = B->N * B->N;
    int inc = 1;

    switch(A->matrix_precision) {
    case single_real:
        // Use BLAS saxpy
        alpha_s = (float)alpha;
        beta_s = (float)beta;
        C_SSCAL(&nElems, &alpha_s, A->matrix, &inc);
        C_SAXPY(&nElems, &beta_s, B->matrix, &inc, A->matrix, &inc);
        break;
    case double_real:
        // Use BLAS daxpy
        C_DSCAL(&nElems, &alpha, A->matrix, &inc);
        C_DAXPY(&nElems, &alpha, B->matrix, &inc, A->matrix, &inc);
        break;
    }
}

/** Matrix addition.
 *
 * A = A + beta * I
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param beta Scalar factor multiplied by A
 */
void bml_add_identity_dense(const bml_matrix_dense_t *A, const double beta)
{
    float beta_s;

    int nElems = A->N * A->N;
    int inc = 1;

    bml_matrix_dense_t *I = bml_identity_matrix_dense(A->matrix_precision, A->N);

    switch(A->matrix_precision) {
    case single_real:
        // Use BLAS saxpy
        beta_s = (float)beta;
        C_SAXPY(&nElems, &beta_s, I->matrix, &inc, A->matrix, &inc);
        break;
    case double_real:
        // Use BLAS daxpy
        C_DAXPY(&nElems, &beta, I->matrix, &inc, A->matrix, &inc);
        break;
    }
}
