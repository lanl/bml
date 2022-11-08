#include "bml.h"
#include "../typed.h"
#include "../macros.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

int TYPED_FUNC(
    test_transpose) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;
    bml_matrix_t *C = NULL;

    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;
    REAL_T *C_dense = NULL;

    int max_row = MIN(N, PRINT_THRESHOLD);
    int max_col = MIN(N, PRINT_THRESHOLD);

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef BML_USE_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    B = bml_transpose_new(A);
    assert(B != NULL);
    C = bml_copy_new(A);
    bml_transpose(C);

    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);
    C_dense = bml_export_to_dense(C, dense_row_major);

    if (bml_getMyRank() == 0)
    {
        LOG_INFO("A:\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
                               0, max_row, 0, max_col);
        LOG_INFO("B:\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense,
                               0, max_row, 0, max_col);
        LOG_INFO("C:\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, C_dense,
                               0, max_row, 0, max_col);
        for (int i = 0; i < N * N; i++)
        {
            if (ABS(B_dense[i] - C_dense[i]) > 1e-12)
            {
                LOG_ERROR
                    ("matrices are not identical B[%d] = %e C[%d] = %e\n", i,
                     B_dense[i], i, C_dense[i]);
                return -1;
            }
        }
        bml_free_memory(A_dense);
        bml_free_memory(B_dense);
        bml_free_memory(C_dense);
    }
    bml_deallocate(&A);
    bml_deallocate(&B);
    bml_deallocate(&C);
    return 0;
}
