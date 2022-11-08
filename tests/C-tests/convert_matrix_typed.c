#include "bml.h"
#include "../typed.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

int TYPED_FUNC(
    test_convert) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A;
    bml_matrix_t *B;
    REAL_T *A_dense;
    REAL_T *B_dense;

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef BML_USE_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    A_dense = bml_allocate_memory(sizeof(REAL_T) * N * N);
    for (int i = 0; i < N * N; i++)
    {
        A_dense[i] = rand() / (double) RAND_MAX;
    }

    A = bml_import_from_dense(matrix_type, matrix_precision, dense_row_major,
                              N, M, A_dense, 0, distrib_mode);

    LOG_DEBUG("test_convert, %d\n", bml_get_N(A));
    B = bml_convert(A, matrix_type, matrix_precision, M, distrib_mode);
    B_dense = bml_export_to_dense(B, dense_row_major);

    if (bml_getMyRank() == 0)
    {
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
                               0, N, 0, N);
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense,
                               0, N, 0, N);
        for (int i = 0; i < N * N; i++)
        {
            if (ABS(A_dense[i] - B_dense[i]) > 1e-12)
            {
                LOG_ERROR("matrix element mismatch A[%d] = %e, B[%d] = %e\n",
                          i, A_dense[i], i, B_dense[i]);
            }
        }
        bml_free_memory(B_dense);
    }
    bml_free_memory(A_dense);
    bml_deallocate(&A);
    bml_deallocate(&B);

    return 0;
}
