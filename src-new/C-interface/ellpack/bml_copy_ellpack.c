#include "../bml_allocate.h"
#include "../bml_copy.h"
#include "../bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_copy_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>

/** Copy an ellpackmatrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_ellpack_t *bml_copy_ellpack(const bml_matrix_ellpack_t *A)
{
    bml_matrix_ellpack_t *B = NULL;

    B = bml_zero_matrix_ellpack(A->matrix_precision, A->N, A->M);
    
    memcpy(B->index, A->index, sizeof(int)*A->N*A->M);
    memcpy(B->nnz, A->nnz, sizeof(int)*A->N);

    switch(B->matrix_precision) {
    case single_precision:
        memcpy(B->value, A->value, sizeof(float)*A->N*A->M);
        break;
    case double_precision:
        memcpy(B->value, A->value, sizeof(double)*A->N*A->M);
        break;
    }
    return B;
}
