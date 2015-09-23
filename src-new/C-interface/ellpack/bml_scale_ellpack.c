#include "../blas.h"
#include "../bml_allocate.h"
#include "../bml_scale.h"
#include "../bml_types.h"
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
bml_matrix_ellpack_t *
bml_scale_ellpack_new(
    const double scale_factor,
    const bml_matrix_ellpack_t * A)
{
    float scale_factor_s;

    bml_matrix_ellpack_t *B = bml_copy_ellpack_new(A);

    int nElems = B->N * B->M;
    int inc = 1;

    switch (B->matrix_precision)
    {
    case single_real:
        scale_factor_s = (float) scale_factor;
        C_SSCAL(&nElems, &scale_factor_s, B->value, &inc);
        break;
    case double_real:
        C_DSCAL(&nElems, &scale_factor, B->value, &inc);
        break;
    }
    return B;
}

/** Scale an ellpack matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void
bml_scale_ellpack(
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B)
{
    float scale_factor_s;

    if (A != B)
        bml_copy_ellpack(A, B);

    int nElems = B->N * B->M;
    int inc = 1;

    switch (A->matrix_precision)
    {
    case single_real:
        scale_factor_s = (float) scale_factor;
        C_SSCAL(&nElems, &scale_factor_s, B->value, &inc);
        break;
    case double_real:
        C_DSCAL(&nElems, &scale_factor, B->value, &inc);
        break;
    }
}
