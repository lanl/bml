#include "../typed.h"
#include "../blas.h"
#include "bml_allocate.h"
#include "bml_scale.h"
#include "bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_copy_ellpack.h"
#include "bml_scale_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>

/** Scale an ellpack matrix - result is a new matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \return A scale version of matrix A.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_scale_ellpack_new) (
    const double scale_factor,
    const bml_matrix_ellpack_t * A)
{
    REAL_T sfactor = (REAL_T) scale_factor;

    bml_matrix_ellpack_t *B = TYPED_FUNC(bml_copy_ellpack_new) (A);

    int nElems = B->N * B->M;
    int inc = 1;

    C_BLAS(SCAL) (&nElems, &sfactor, B->value, &inc);

    return B;
}

/** Scale an ellpack matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void TYPED_FUNC(
    bml_scale_ellpack) (
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B)
{
    REAL_T sfactor = (REAL_T) scale_factor;

    if (A != B)
        TYPED_FUNC(bml_copy_ellpack) (A, B);

    int nElems = B->N * B->M;
    int inc = 1;

    C_BLAS(SCAL) (&nElems, &sfactor, B->value, &inc);
}

void TYPED_FUNC(
    bml_scale_inplace_ellpack) (
    const double scale_factor,
    bml_matrix_ellpack_t * A)
{
    REAL_T scale_factor_ = (REAL_T) scale_factor;
    int number_elements = A->N * A->N;
    int inc = 1;

    C_BLAS(SCAL) (&number_elements, &scale_factor_, A->matrix, &inc);
}
