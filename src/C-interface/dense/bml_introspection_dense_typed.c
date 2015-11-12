#include "../macros.h"
#include "../typed.h"
#include "bml_introspection_dense.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>

/** Return the bandwidth of a row in the matrix.
 *
 * \param A The bml matrix.
 * \param i The row index.
 * \return The bandwidth of row i.
 */
int TYPED_FUNC(
    bml_get_row_bandwidth_dense) (
    const bml_matrix_dense_t * A,
    const int i)
{
    assert(A != NULL);

    REAL_T *A_matrix = A->matrix;
    int bandwidth = 0;

    for (int j = 0; j < A->N; j++)
    {
        if (is_above_threshold(A_matrix[ROWMAJOR(i, j, A->N)], 0))
        {
            bandwidth++;
        }
    }
    return bandwidth;
}
