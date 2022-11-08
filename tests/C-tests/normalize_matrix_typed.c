#include "bml.h"
#include "../typed.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define REL_TOL 2e-6
#else
#define REL_TOL 1e-12
#endif

int TYPED_FUNC(
    test_normalize) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;
    double *A_gbnd = NULL;
    double *B_gbnd = NULL;
    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef BML_USE_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    REAL_T scale_factor = 2.5;
    double threshold = 0.0;

    A = bml_identity_matrix(matrix_type, matrix_precision, N, M,
                            distrib_mode);
    bml_scale_inplace(&scale_factor, A);
    A_gbnd = bml_gershgorin(A);

    A_dense = bml_export_to_dense(A, dense_row_major);

    if (bml_getMyRank() == 0)
        A_dense[0] = scale_factor * scale_factor;
    B = bml_import_from_dense(matrix_type, matrix_precision, dense_row_major,
                              N, M, A_dense, threshold, distrib_mode);

    B_gbnd = bml_gershgorin(B);

    if (bml_getMyRank() == 0)
    {
        LOG_INFO("B_gbnd=%le,%le\n", B_gbnd[0], B_gbnd[1]);
        bml_free_memory(A_dense);
    }
    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);
    if (bml_getMyRank() == 0)
    {
        LOG_INFO("A\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense,
                               0, N, 0, N);
        LOG_INFO("B\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense,
                               0, N, 0, N);
    }

    bml_normalize(B, B_gbnd[0], B_gbnd[1]);
    if (bml_getMyRank() == 0)
        bml_free_memory(B_dense);
    B_dense = bml_export_to_dense(B, dense_row_major);
    if (bml_getMyRank() == 0)
    {
        LOG_INFO("Normalized B\n");
        bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense,
                               0, N, 0, N);
    }
#ifdef BML_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // A is a diagonal matrix with uniform values "scale_factor" on diagonal
    if ((fabs(A_gbnd[1] - REAL_PART(scale_factor))) > REL_TOL
        || (A_gbnd[1] - A_gbnd[0]) > REL_TOL)
    {
        LOG_ERROR
            ("A: incorrect maxeval or maxminusmin; mineval = %e maxeval = %e maxminusmin = %e\n",
             A_gbnd[0], A_gbnd[1], A_gbnd[1] - A_gbnd[0]);
        return -1;
    }

    if ((fabs(B_gbnd[1] - REAL_PART(scale_factor * scale_factor))) > REL_TOL
        ||
        (fabs
         ((B_gbnd[1] - B_gbnd[0]) -
          REAL_PART(scale_factor * scale_factor - scale_factor))) > REL_TOL)
    {
        LOG_ERROR
            ("B: incorrect maxeval or maxminusmin; mineval = %e maxeval = %e maxminusmin = %e\n",
             B_gbnd[0], B_gbnd[1], B_gbnd[1] - B_gbnd[0]);
        return -1;
    }

    if (bml_getMyRank() == 0)
        if (ABS(B_dense[0]) > REL_TOL)
        {
            LOG_ERROR
                ("normalize error, incorrect B[0] = %e instead of 0\n",
                 B_dense[0]);
            return -1;
        }

    LOG_INFO("normalize matrix test passed\n");

    if (bml_getMyRank() == 0)
    {
        bml_free_memory(A_dense);
        bml_free_memory(B_dense);
    }
    bml_free_memory(A_gbnd);
    bml_free_memory(B_gbnd);
    bml_deallocate(&A);
    bml_deallocate(&B);

    return 0;
}
