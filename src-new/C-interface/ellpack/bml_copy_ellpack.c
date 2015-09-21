#include "../bml_allocate.h"
#include "../bml_copy.h"
#include "../bml_types.h"
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
bml_matrix_ellpack_t *
bml_copy_ellpack (const bml_matrix_ellpack_t * A)
{
    bml_matrix_ellpack_t *B = NULL;

    B = bml_zero_matrix_ellpack (A->matrix_precision, A->N, A->M);

    memcpy (B->index, A->index, sizeof (int) * A->N * A->M);
    memcpy (B->nnz, A->nnz, sizeof (int) * A->N);

    switch (B->matrix_precision)
    {
    case single_real:
        memcpy (B->value, A->value, sizeof (float) * A->N * A->M);
        break;
    case double_real:
        memcpy (B->value, A->value, sizeof (double) * A->N * A->M);
        break;
    }
    return B;
}

/** Copy an ellpack matrix - result is an existing matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A
 */
void bml_copy_ellpack(const bml_matrix_ellpack_t *A, const bml_matrix_ellpack_t *B)
{
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
}
