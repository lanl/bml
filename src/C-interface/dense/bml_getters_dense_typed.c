#include "../macros.h"
#include "../typed.h"
#include "../bml_introspection.h"
#include "bml_getters_dense.h"
#include "bml_types_dense.h"

#include <complex.h>


// Getter for diagonal

void TYPED_FUNC(
    bml_get_diagonal_dense) (
    bml_matrix_dense_t * A,
    REAL_T * diagonal)
{
    int N = bml_get_N(A);
    REAL_T *A_matrix = A->matrix;

    for (int j = 0; j < N; j++)
    {
        diagonal[j] = A_matrix[ROWMAJOR(j, j, N, N)];
    }
}


// Getter for row

void TYPED_FUNC(
    bml_get_row_dense) (
    bml_matrix_dense_t * A,
    const int i,
    REAL_T * row)
{
    int N = bml_get_N(A);
    REAL_T *A_matrix = A->matrix;

    for (int j = 0; j < N; j++)
    {
        row[j] = A_matrix[ROWMAJOR(i, j, N, N)];
    }
}
