#include "../blas.h"
#include "../bml_allocate.h"
#include "../bml_multiply.h"
#include "../bml_types.h"
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
void bml_multiply_dense(const bml_matrix_dense_t *A, const bml_matrix_dense_t *B, const bml_matrix_dense_t *C, const double alpha, const double beta)
{
    float alpha_s, beta_s;
    char trans = 'N';

    int hdim = A->N;

    switch(A->matrix_precision) {
    case single_real:
        // Use BLAS sgemm
        alpha_s = (float)alpha;
        beta_s = (float)beta;
        C_SGEMM(&trans, &trans, &hdim, &hdim, &hdim, &alpha_s, A->matrix, &hdim, B->matrix, &hdim, &beta_s, C->matrix, &hdim);
        break;
    case double_real:
        // Use BLAS dgemm
        C_DGEMM(&trans, &trans, &hdim, &hdim, &hdim, &alpha, A->matrix, &hdim, B->matrix, &hdim, &beta, C->matrix, &hdim);
        break;
    }
}
