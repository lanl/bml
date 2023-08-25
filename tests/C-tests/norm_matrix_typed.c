#include "bml.h"
#include "../typed.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define REL_TOL 1e-5
#else
#define REL_TOL 1e-8
#endif

int TYPED_FUNC(
    test_norm) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;
    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef BML_USE_MPI
    if (bml_getNRanks() > 1)
    {
        if (bml_getMyRank() == 0)
            LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif
    if (bml_getMyRank() == 0)
        LOG_INFO("N = %d\n", N);

    const double alpha = 1.2;
    const double beta = 0.8;
    const double threshold = 0.0;

    double sum = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;
    double fnorm = 0.0;
    double sqrt_sum = 0.0;
    double sqrt_sum2 = 0.0;

    const double tol = REL_TOL * N * M;

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    B = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);

    if (bml_getMyRank() == 0)
        LOG_INFO("Matrices:\n");
    bml_print_bml_matrix(A, 0, N, 0, N);
    bml_print_bml_matrix(B, 0, N, 0, N);

    if (bml_getMyRank() == 0)
        LOG_INFO("Export to dense...\n");
    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);

    sum2 = bml_sum_squares2(A, B, alpha, beta, threshold);

    sum3 = bml_sum_AB(A, B, alpha, threshold);

    if (bml_getMyRank() == 0)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                sum4 += alpha * A_dense[i * N + j] * B_dense[i * N + j];
            }
        }
    }

    bml_add(A, B, alpha, beta, threshold);
    sum = bml_sum_squares(A);
    fnorm = bml_fnorm(A);

    if (bml_getMyRank() == 0)
    {
        LOG_INFO("tol = %le\n", tol);
        LOG_INFO("sum = %le\n", sum);
        LOG_INFO("sum2 = %le\n", sum2);
        LOG_INFO("sum3 = %le\n", sum3);
        LOG_INFO("sum4 = %le\n", sum4);
        LOG_INFO("fnorm = %le\n", fnorm);
    }

    if (bml_getMyRank() == 0)
        if (fabs(sum3 - sum4) > tol)
        {
            LOG_ERROR
                ("incorrect product of matrix A and B; sum3 = %e sum4 = %e\n",
                 sum3, sum4);
            return -1;
        }

    if (fabs(sum - sum2) > tol)
    {
        LOG_ERROR
            ("incorrect sum of squares or sum of squares2; sum = %e sum2 = %e\n",
             sum, sum2);
        return -1;
    }

    sqrt_sum = sqrt(sum);
    sqrt_sum2 = sqrt(sum2);

    if ((fabs(sqrt_sum - sqrt_sum2) > tol) ||
        (fabs(sqrt_sum - fnorm) > tol) || (fabs(sqrt_sum2 - fnorm) > tol))
    {
        LOG_ERROR
            ("incorrect sqrt(sum) or sqrt(sum2) of fnorm ; sqrt_sum = %e sqrt_sum2 = %e fnorm = %e\n",
             sqrt_sum, sqrt_sum2, fnorm);

        return -1;
    }

    LOG_INFO("norm matrix test passed\n");
    if (bml_getMyRank() == 0)
    {
        bml_free_memory(A_dense);
        bml_free_memory(B_dense);
    }
    bml_deallocate(&A);
    bml_deallocate(&B);

    return 0;
}
