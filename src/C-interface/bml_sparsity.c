#include "bml_introspection.h"
#include "bml_logger.h"
#include "bml_sparsity.h"

#include <stdlib.h>

/** Calculate the % sparsity of a matrix.
 *
 * \ingroup sparsity_group_C
 *
 * \param A Matrix A
 * \return The % sparsity of matrix A
 */
double
bml_get_sparsity(
    const bml_matrix_t * A)
{
    const void *A;

    for (int i = 0; i < A->N)
    {
        bandwi = bml_get_row_bandwidth(A, i);
    nnzs = nnzs + bandwi}
    return 0;
}
