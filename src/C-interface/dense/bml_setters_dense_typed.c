#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#endif

#include "../../macros.h"
#include "../../typed.h"
#include "../bml_introspection.h"
#include "../bml_logger.h"
#include "../bml_utilities.h"
#include "bml_setters_dense.h"
#include "bml_types_dense.h"
#include "bml_allocate_dense.h"

#include <complex.h>
#include <stdio.h>

void TYPED_FUNC(
    bml_set_element_dense) (
    bml_matrix_dense_t * A,
    int i,
    int j,
    void *value)
{
    int N = bml_get_N(A);

    if (N < 0)
    {
        LOG_ERROR("A is not intialized\n");
    }

#ifdef BML_USE_MAGMA
    MAGMA(setvector) (N, (MAGMA_T *) value, 1,
                      (MAGMA_T *) A->matrix + i * A->ld + j, 1, bml_queue());
#else
    REAL_T *A_matrix = A->matrix;
#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
    A_matrix[ROWMAJOR(i, j, N, N)] = *((REAL_T *) value);
#ifdef MKL_GPU
// push back to GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
#endif
}

void TYPED_FUNC(
    bml_set_row_dense) (
    bml_matrix_dense_t * A,
    int i,
    void *_row)
{
    REAL_T *row = _row;
    int N = bml_get_N(A);

    if (N < 0)
    {
        LOG_ERROR("A is not intialized\n");
    }

#ifdef BML_USE_MAGMA
    MAGMA(setvector) (N, (MAGMA_T *) row, 1,
                      (MAGMA_T *) A->matrix + i * A->ld, 1, bml_queue());
#else
    REAL_T *A_matrix = A->matrix;
#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
    for (int j = 0; j < N; j++)
    {
        A_matrix[ROWMAJOR(i, j, N, N)] = row[j];
    }
#ifdef MKL_GPU
// push back to GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
#endif
}

void TYPED_FUNC(
    bml_set_diagonal_dense) (
    bml_matrix_dense_t * A,
    void *_diagonal)
{
    REAL_T *diagonal = _diagonal;
    int N = bml_get_N(A);

    if (N < 0)
    {
        LOG_ERROR("A is not intialized\n");
    }

#ifdef BML_USE_MAGMA
    MAGMA_T *diagonal_ = malloc(N * sizeof(MAGMA_T));
    for (int j = 0; j < N; j++)
    {
        diagonal_[j] = MAGMACOMPLEX(MAKE) (diagonal[j], 0);
    }
    MAGMA(setvector) (N, diagonal_, 1, (MAGMA_T *) A->matrix, A->ld + 1,
                      bml_queue());
    free(diagonal_);
#else
    REAL_T *A_matrix = A->matrix;
#ifdef MKL_GPU
// pull from GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
    for (int j = 0; j < N; j++)
    {
        A_matrix[ROWMAJOR(j, j, N, N)] = diagonal[j];
    }
#ifdef MKL_GPU
// push to GPU
#pragma omp target update to(A_matrix[0:N*N])
#endif
#endif
}
