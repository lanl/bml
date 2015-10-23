#include "bml.h"
#include "bml_test.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

int
test_function(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A;
    REAL_T *A_dense;
    REAL_T *B_dense;

    A_dense = bml_allocate_memory(sizeof(REAL_T) * N * N);
    for (int i = 0; i < N * N; i++)
    {
        A_dense[i] = rand() / (double) RAND_MAX;
    }
    A = bml_convert_from_dense(matrix_type, matrix_precision, N, A_dense,
                               0, M);
    B_dense = bml_convert_to_dense(A);
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(A_dense[i] - B_dense[i]) > 1e-12)
        {
            bml_print_dense_matrix(N, matrix_precision, A_dense, 0, N, 0, N);
            bml_print_dense_matrix(N, matrix_precision, B_dense, 0, N, 0, N);
            LOG_ERROR("matrix element mismatch\n");
        }
    }
    bml_free_memory(A_dense);
    bml_free_memory(B_dense);
    bml_deallocate(&A);
    return 0;
}
