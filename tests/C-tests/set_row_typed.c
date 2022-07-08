#include "bml.h"
#include "../typed.h"
#include "bml_getters.h"
#include "bml_setters.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int TYPED_FUNC(
    test_set_row) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    REAL_T *A_row = NULL;
    REAL_T *B_row = NULL;

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef DO_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    // Create a row
    switch (matrix_precision)
    {
        case single_real:
            A_row = calloc(N, sizeof(float));
            break;
        case double_real:
            A_row = calloc(N, sizeof(double));
            break;
#ifdef BML_COMPLEX
        case single_complex:
            A_row = calloc(N, sizeof(float complex));
            break;
        case double_complex:
            A_row = calloc(N, sizeof(double complex));
            break;
#endif
        default:
            LOG_DEBUG("matrix_precision is not set");
            break;
    }

    for (int i = 0; i < N; i++)
    {
        A_row[i] = (REAL_T) i;
    }
    for (int i = 0; i < N; i++)
        LOG_INFO("A_row[%d]=%e\n", i, A_row[i]);
    A = bml_zero_matrix(matrix_type, matrix_precision, N, M, distrib_mode);

    // Set the 9th row
    int irow = 9;
    bml_set_row(A, irow, A_row, 1.0e-10);
    bml_print_bml_matrix(A, 0, N, 0, N);

    // Retrieve the second row
    B_row = bml_get_row(A, irow);

    // Verify B_row is correct
    for (int i = 0; i < N; i++)
        LOG_INFO("A_row[%d]=%e, B_row[%d]=%e\n", i, i, A_row[i], B_row[i]);

    for (int i = 0; i < N; i++)
    {
        if (ABS(A_row[i] - B_row[i]) > 1e-12)
        {
            LOG_ERROR("A_row[%d]=%le, B_row[%d]=%le\n", i, i, A_row[i],
                      B_row[i]);
            LOG_ERROR
                ("bml_set_row and/or bml_get_row failed for task %d\n",
                 bml_getMyRank());
            return -1;
        }
    }

    bml_deallocate(&A);
    free(A_row);
    free(B_row);

    LOG_INFO("bml_set_row passed for task %d\n", bml_getMyRank());

    return 0;
}
