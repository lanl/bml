#include "../macros.h"
#include "../typed.h"
#include "../bml_introspection.h"
#include "bml_setters_dense.h"
#include "bml_types_dense.h"

#include <complex.h>

void TYPED_FUNC(
    bml_set_row_dense) (
    bml_matrix_dense_t * A,
    const int i,
    const REAL_T * row)
{
    int N = bml_get_N(A);
    REAL_T *A_matrix = A->matrix;

    for (int j = 0; j < N; j++)
    {
        A_matrix[ROWMAJOR(i, j, N, N)] = row[j];
    }
}
