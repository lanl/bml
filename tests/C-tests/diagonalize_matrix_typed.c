#include "bml.h"
#include "../typed.h"
#include "../macros.h"
#include "../C-interface/dense/bml_getters_dense.h"
#include "../C-interface/bml_logger.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define REL_TOL 1.2e-5
#else
#define REL_TOL 1e-11
#endif

int TYPED_FUNC(
    test_diagonalize) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *A_t = NULL;
    REAL_T *eigenvalues = NULL;
    bml_matrix_t *eigenvectors = NULL;
    bml_matrix_t *ct = NULL;
    bml_matrix_t *aux = NULL;
    bml_matrix_t *aux1 = NULL;
    bml_matrix_t *aux2 = NULL;
    bml_matrix_t *id = NULL;
    float fnorm;

    int max_row = MIN(N, PRINT_THRESHOLD);
    int max_col = MIN(N, PRINT_THRESHOLD);

    LOG_INFO("rel. tolerance = %e\n", REL_TOL);

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef BML_USE_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);

    //LOG_INFO("A = \n");
    //bml_print_bml_matrix(A, 0, max_row, 0, max_col);

    A_t = bml_transpose_new(A);

    //LOG_INFO("A_t = \n");
    //bml_print_bml_matrix(A_t, 0, max_row, 0, max_col);

    bml_add(A, A_t, 0.5, 0.5, 0.0);

    LOG_INFO("(A + A_t)/2 = \n");
    bml_print_bml_matrix(A, 0, max_row, 0, max_col);

    switch (matrix_precision)
    {
        case single_real:
            eigenvalues = bml_allocate_memory(N * sizeof(float));
#ifdef INTEL_OPT
#pragma omp parallel for simd
#pragma vector aligned
            for (int i = 0; i < N; i++)
            {
                __assume_aligned(eigenvalues, 64);
                eigenvalues[i] = 0.0;
            }
#endif
            break;
        case double_real:
            eigenvalues = bml_allocate_memory(N * sizeof(double));
#ifdef INTEL_OPT
#pragma omp parallel for simd
#pragma vector aligned
            for (int i = 0; i < N; i++)
            {
                __assume_aligned(eigenvalues, 64);
                eigenvalues[i] = 0.0;
            }
#endif
            break;
#ifdef BML_COMPLEX
        case single_complex:
            eigenvalues = bml_allocate_memory(N * sizeof(float complex));
#ifdef INTEL_OPT
#pragma omp parallel for simd
#pragma vector aligned
            for (int i = 0; i < N; i++)
            {
                __assume_aligned(eigenvalues, 64);
                eigenvalues[i] = 0.0;
            }
#endif
            break;
        case double_complex:
            eigenvalues = bml_allocate_memory(N * sizeof(double complex));
#ifdef INTEL_OPT
#pragma omp parallel for simd
#pragma vector aligned
            for (int i = 0; i < N; i++)
            {
                __assume_aligned(eigenvalues, 64);
                eigenvalues[i] = 0.0;
            }
#endif
            break;
#endif
        default:
            LOG_DEBUG("matrix_precision is not set");
            break;
    }

    eigenvectors = bml_zero_matrix(matrix_type, matrix_precision,
                                   N, M, distrib_mode);

    aux = bml_zero_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    aux1 = bml_zero_matrix(matrix_type, matrix_precision, N, M, distrib_mode);
    aux2 = bml_zero_matrix(matrix_type, matrix_precision, N, M, distrib_mode);

    bml_diagonalize(A, eigenvalues, eigenvectors);

    if (bml_getMyRank() == 0)
    {
        LOG_INFO("%s\n", "eigenvectors");
    }
    bml_print_bml_matrix(eigenvectors, 0, max_row, 0, max_col);
    if (bml_getMyRank() == 0)
    {
        LOG_INFO("%s\n", "eigenvalues");
        for (int i = 0; i < max_row; i++)
            LOG_INFO("val = %e  i%e\n", REAL_PART(eigenvalues[i]),
                     IMAGINARY_PART(eigenvalues[i]));
    }

    ct = bml_transpose_new(eigenvectors);
    if (bml_getMyRank() == 0)
    {
        LOG_INFO("%s\n", "transpose eigenvectors");
    }
    bml_print_bml_matrix(ct, 0, max_row, 0, max_col);

    bml_multiply(ct, eigenvectors, aux2, 1.0, 0.0, 0.0);        // C^t*C

    if (bml_getMyRank() == 0)
        LOG_INFO("C^t*C matrix:\n");
    bml_print_bml_matrix(aux2, 0, max_row, 0, max_col);
    REAL_T *aux2_dense = bml_export_to_dense(aux2, dense_row_major);
    if (bml_getMyRank() == 0)
    {
        LOG_INFO("%s\n", "check eigenvectors norms");
        for (int i = 0; i < N; i++)
        {
            REAL_T val = aux2_dense[i + N * i];
            if (ABS(val - (REAL_T) 1.0) > REL_TOL)
            {
                LOG_INFO("i = %d, val = %e  i%e\n", i, REAL_PART(val),
                         IMAGINARY_PART(val));
                LOG_ERROR
                    ("Error in matrix diagonalization; eigenvector not normalized\n");
            }
        }
        bml_free_memory(aux2_dense);
    }

    id = bml_identity_matrix(matrix_type, matrix_precision, N, M,
                             distrib_mode);
    if (bml_getMyRank() == 0)
        LOG_INFO("Identity matrix:\n");
    bml_print_bml_matrix(id, 0, max_row, 0, max_col);

    bml_add(aux2, id, 1.0, -1.0, 0.0);
    if (bml_getMyRank() == 0)
        LOG_INFO("C^txC^t-Id matrix:\n");
    bml_print_bml_matrix(aux2, 0, max_row, 0, max_col);
    fnorm = bml_fnorm(aux2);
    if (fabsf(fnorm) > N * REL_TOL)
    {
        LOG_ERROR
            ("Error in matrix diagonalization; fnorm(C^txC^t-Id) = %e\n",
             fnorm);
        return -1;
    }
    bml_set_diagonal(aux1, eigenvalues, 0.0);
    if (bml_getMyRank() == 0)
        LOG_INFO("Matrix after setting diagonal:\n");
    bml_print_bml_matrix(aux1, 0, max_row, 0, max_col);

    bml_multiply(aux1, ct, aux2, 1.0, 0.0, 0.0);        // D*C^t
    bml_multiply(eigenvectors, aux2, aux, 1.0, 0.0, 0.0);       // C*(D*C^t)

    if (bml_getMyRank() == 0)
        LOG_INFO("C*(D*C^t) matrix:\n");
    bml_print_bml_matrix(aux, 0, max_row, 0, max_col);

    bml_add(aux, A, 1.0, -1.0, 0.0);
    if (bml_getMyRank() == 0)
        LOG_INFO("C*(D*C^t)-A matrix:\n");
    bml_print_bml_matrix(aux, 0, max_row, 0, max_col);

    fnorm = bml_fnorm(aux);

    if (fabsf(fnorm) > N * REL_TOL || (fnorm != fnorm))
    {
        LOG_ERROR
            ("Error in matrix diagonalization; fnorm(CDC^t-A) = %e\n", fnorm);
        return -1;
    }

    bml_deallocate(&A);
    bml_deallocate(&aux);
    bml_deallocate(&aux1);
    bml_deallocate(&aux2);
    bml_deallocate(&ct);
    bml_deallocate(&A_t);
    bml_deallocate(&eigenvectors);
    bml_deallocate(&id);
    bml_free_memory(eigenvalues);

    LOG_INFO("diagonalize matrix test passed\n");

    return 0;
}
