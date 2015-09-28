#include "../typed.h"
#include "bml_allocate.h"
#include "bml_copy.h"
#include "bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_copy_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>

/** Copy an ellpack matrix - result is a new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_copy_ellpack_new) (
    const bml_matrix_ellpack_t * A)
{
    bml_matrix_ellpack_t *B =
        TYPED_FUNC(bml_zero_matrix_ellpack) (A->N, A->M);

    memcpy(B->index, A->index, sizeof(int) * A->N * A->M);
    memcpy(B->nnz, A->nnz, sizeof(int) * A->N);

    memcpy(B->value, A->value, sizeof(REAL_T) * A->N * A->M);

    return B;
}

/** Copy an ellpack matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A
 */
void TYPED_FUNC(
    bml_copy_ellpack) (
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B)
{

    memcpy(B->index, A->index, sizeof(int) * A->N * A->M);
    memcpy(B->nnz, A->nnz, sizeof(int) * A->N);

    memcpy(B->value, A->value, sizeof(REAL_T) * A->N * A->M);
}
