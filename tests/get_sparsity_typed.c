#include "bml.h"
#include "typed.h"
#include "bml_introspection.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int TYPED_FUNC(
    test_get_sparsity) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    double sparsity;
    double sparsity_ref;

    A = bml_identity_matrix(matrix_type, matrix_precision, N, M, sequential);

    sparsity = bml_get_sparsity(A, 0.5);

    sparsity_ref = 1.0 - (double) N / (double) (N * N);

    if (ABS(sparsity - sparsity_ref) > 1e-12)
    {
        LOG_ERROR("bml_get_sparsity is corrupted\n");
        return -1;
    }

    printf("Sparsity = %f\n", sparsity);

    bml_deallocate(&A);

    LOG_INFO("bml_get_sparsity passed\n");

    return 0;
}
