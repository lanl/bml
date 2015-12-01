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
    bml_matrix_t *A = NULL;
    bml_matrix_t *B = NULL;
    double *A_gbnd = NULL;
    double *B_gbnd = NULL;
    REAL_T *A_dense = NULL;
    REAL_T *B_dense = NULL;

    double scale_factor = 2.5;
    double threshold = 0.0;

    A = bml_identity_matrix(matrix_type, matrix_precision, N, M);
    bml_scale_inplace(scale_factor, A);
    A_gbnd = bml_gershgorin(A);

    A_dense = bml_export_to_dense(A, dense_row_major);
    A_dense[0] = scale_factor * scale_factor;
    B = bml_import_from_dense(matrix_type, matrix_precision, dense_row_major,
                              N, A_dense, threshold, M);
    B_gbnd = bml_gershgorin(B);

    bml_free_memory(A_dense);
    A_dense = bml_export_to_dense(A, dense_row_major);
    B_dense = bml_export_to_dense(B, dense_row_major);
    bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense, 0,
                           N, 0, N);
    bml_print_dense_matrix(N, matrix_precision, dense_row_major, B_dense, 0,
                           N, 0, N);

    if ((fabs(A_gbnd[0] - scale_factor)) > 1e-12 || 
         A_gbnd[1] > 1e-12)
    {
        LOG_ERROR
            ("incorrect maxeval or maxminusmin; maxeval = %e maxminusmin = %e\n",
             A_gbnd[0], A_gbnd[1]);
        return -1;
    }

    if ((fabs(B_gbnd[0] - scale_factor*scale_factor)) > 1e-12 || 
        (fabs(B_gbnd[1] - (scale_factor*scale_factor - scale_factor))) > 1e-12)
    {
        LOG_ERROR
            ("incorrect maxeval or maxminusmin; maxeval = %e maxminusmin = %e\n",
             B_gbnd[0], B_gbnd[1]);
        return -1;
    }

    LOG_INFO("gershgorin matrix test passed\n");
    bml_free_memory(A_dense);
    bml_free_memory(B_dense);
    bml_deallocate(&A);
    bml_deallocate(&B);

    return 0;
}
