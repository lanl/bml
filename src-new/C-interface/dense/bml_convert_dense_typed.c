#include "../typed.h"
#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_convert.h"
#include "bml_convert_dense.h"
#include "bml_logger.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns
 * \param matrix_precision The real precision
 * \param A The dense matrix
 * \return The bml matrix
 */
bml_matrix_dense_t *
TYPED_FUNC(bml_convert_from_dense_dense)(
    const int N,
    const void *A)
{
    bml_matrix_dense_t *A_bml = TYPED_FUNC(bml_zero_matrix_dense)(N);

    memcpy(A_bml->matrix, A, sizeof(REAL_T) * N * N);
    return A_bml;
}

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *
TYPED_FUNC(bml_convert_to_dense_dense)(
    const bml_matrix_dense_t * A)
{
    REAL_T *A_dense = bml_allocate_memory(sizeof(REAL_T) * A->N * A->N);


    memcpy(A_dense, A->matrix, sizeof(REAL_T) * A->N * A->N);

    return A_dense;
}
