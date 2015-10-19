#include "../typed.h"
#include "bml_allocate.h"
#include "bml_copy.h"
#include "bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

/** Copy a dense matrix - result in new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_dense_t *TYPED_FUNC(
    bml_copy_dense_new) (
    const bml_matrix_dense_t * A)
{
    bml_matrix_dense_t *B = TYPED_FUNC(bml_zero_matrix_dense)(A->N);
    memcpy(B->matrix, A->matrix, sizeof(REAL_T) * A->N * A->N);
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
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B)
{
    memcpy(B->matrix, A->matrix, sizeof(REAL_T) * A->N * A->N);
}
