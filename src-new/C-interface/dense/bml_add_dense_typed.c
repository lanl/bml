#include "../typed.h"
#include "../blas.h"
#include "bml_allocate.h"
#include "bml_add.h"
#include "bml_types.h"
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
void TYPED_FUNC(
    bml_add_dense) (
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta)
{
    REAL_T salpha = (REAL_T) alpha;
    REAL_T sbeta = (REAL_T) beta;
    int nElems = B->N * B->N;
    int inc = 1;

    // Use BLAS saxpy
    C_BLAS(SCAL) (&nElems, &salpha, A->matrix, &inc);
    C_BLAS(AXPY) (&nElems, &sbeta, B->matrix, &inc, A->matrix, &inc);
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
void TYPED_FUNC(
    bml_add_identity_dense) (
    const bml_matrix_dense_t * A,
    const double beta)
{
    REAL_T sbeta = (REAL_T) beta;
    int nElems = A->N * A->N;
    int inc = 1;

    bml_matrix_dense_t *I = TYPED_FUNC(bml_identity_matrix_dense) (A->N);

    C_BLAS(AXPY) (&nElems, &sbeta, I->matrix, &inc, A->matrix, &inc);

    bml_deallocate_dense(I);
}
