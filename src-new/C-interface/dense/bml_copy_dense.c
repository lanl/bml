#include "../bml_allocate.h"
#include "../bml_copy.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_types_dense.h"

#include <stdlib.h>
#include <string.h>

/** Copy a dense matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_dense_t *
bml_copy_dense (const bml_matrix_dense_t * A)
{
    bml_matrix_dense_t *B = NULL;

    B = bml_zero_matrix_dense (A->matrix_precision, A->N);

    switch (A->matrix_precision)
    {
    case single_real:
        memcpy (B->matrix, A->matrix, sizeof (float) * A->N * A->N);
        break;
    case double_real:
        memcpy (B->matrix, A->matrix, sizeof (double) * A->N * A->N);
        break;
    }
    return B;
}
