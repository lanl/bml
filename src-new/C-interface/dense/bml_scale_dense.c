#include "../blas.h"
#include "../bml_allocate.h"
#include "../bml_scale.h"
#include "../bml_types.h"
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
bml_matrix_dense_t *bml_scale_dense_new(const double scale_factor, const bml_matrix_dense_t *A)
{
    float scale_factor_s;

    bml_matrix_dense_t *B = NULL;

    B = bml_copy_dense_new(A);

    int nElems = B->N * B->N;
    int inc = 1;

    switch(A->matrix_precision) {
    case single_real:
        // Use BLAS sscal
        scale_factor_s = (float)scale_factor;
        C_SSCAL(&nElems, &scale_factor_s, B->matrix, &inc);
        break;
    case double_real:
        // Use BLAS dscal
        C_DSCAL(&nElems, &scale_factor, B->matrix, &inc);
        break;
    }
    return B;
}

/** Scale a dense matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void bml_scale_dense(const double scale_factor, const bml_matrix_dense_t *A, const bml_matrix_dense_t *B)
{
    float scale_factor_s;

    if (A != B) bml_copy_dense(A, B);

    int nElems = B->N * B->N;
    int inc = 1;

    switch(A->matrix_precision) {
    case single_real:
        // Use BLAS sscal
        scale_factor_s = (float)scale_factor;
        C_SSCAL(&nElems, &scale_factor_s, B->matrix, &inc);
        break;
    case double_real:
        // Use BLAS dscal
        C_DSCAL(&nElems, &scale_factor, B->matrix, &inc);
        break;
    }
}
