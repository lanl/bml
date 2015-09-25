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
void TYPED_FUNC(bml_multiply_dense) (const bml_matrix_dense_t *A, const bml_matrix_dense_t *B, const bml_matrix_dense_t *C, const double alpha, const double beta)
{
    REAL_T salpha = (REAL_T)alpha;
    REAL_T sbeta = (REAL_T)beta;
    char trans = 'N';

    int hdim = A->N;

    C_BLAS(GEMM)(&trans, &trans, &hdim, &hdim, &hdim, &salpha, A->matrix, &hdim, B->matrix, &hdim, &sbeta, C->matrix, &hdim);
}
