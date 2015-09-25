#include "../typed.h"
#include "../blas.h"
#include "bml_allocate.h"
#include "bml_scale.h"
#include "bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_scale_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Scale a dense matrix - result in new matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \return A scaled version of matrix A.
 */
bml_matrix_dense_t *
TYPED_FUNC(bml_scale_dense_new) (
    const double scale_factor,
    const bml_matrix_dense_t * A)
{
    REAL_T sfactor = (REAL_T)scale_factor;

    bml_matrix_dense_t *B = TYPED_FUNC(bml_copy_dense_new)(A);
    int nElems = B->N * B->N;
    int inc = 1;

    // Use BLAS sscal/dscal
    C_SSCAL(&nElems, &sfactor, B->matrix, &inc);
        //C_DSCAL(&nElems, &scale_factor, B->matrix, &inc);
        
    return B;
}

/** Scale a dense matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void
TYPED_FUNC(bml_scale_dense) (
    const double scale_factor,
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B)
{
    REAL_T sfactor = scale_factor;

    if (A != B)
        TYPED_FUNC(bml_copy_dense)(A, B);

    int nElems = B->N * B->N;
    int inc = 1;

    C_SSCAL(&nElems, &sfactor, B->matrix, &inc);
        //C_DSCAL(&nElems, &scale_factor, B->matrix, &inc);
}
