#include "../bml_allocate.h"
#include "../bml_scale.h"
#include "../bml_types.h"
#include "bml_allocate_ellpack.h"
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
bml_matrix_ellpack_t *bml_scale_ellpack_new(const double scale_factor, const bml_matrix_ellpack_t *A)
{
    bml_matrix_ellpack_t *B = NULL;

    B = bml_zero_matrix_ellpack(A->matrix_precision, A->N, A->M);
    
    memcpy(B->index, A->index, sizeof(int)*A->N*A->M);
    memcpy(B->nnz, A->nnz, sizeof(int)*A->N);

    switch(B->matrix_precision) {
    case single_real:
        memcpy(B->value, A->value, sizeof(float)*A->N*A->M);
        break;
    case double_real:
        memcpy(B->value, A->value, sizeof(double)*A->N*A->M);
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
void bml_scale_ellpack(const double scale_factor, const bml_matrix_ellpack_t *A, bml_matrix_ellpack_t *B)
{
    switch(A->matrix_precision) {
    case single_real:
        break;
    case double_real:
        break;
    }
}
