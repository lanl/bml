#include "bml.h"
#include "bml_test.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>

#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define REL_TOL 1e-6
#else
#define REL_TOL 1e-12
#endif

int
test_function(
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    bml_matrix_t *A_t = NULL;

    double *eigenvalues = NULL;
    bml_matrix_t *eigenvectors = NULL;

    LOG_DEBUG("rel. tolerance = %e\n", REL_TOL);

    A = bml_random_matrix(matrix_type, matrix_precision, N, M);
    A_t = bml_transpose_new(A);
    bml_add(A, A_t, 0.5, 0.5, 0.0);
    bml_print_bml_matrix(A, 0, N, 0, N);
    eigenvalues = calloc(N, sizeof(double));
    eigenvectors = bml_zero_matrix(matrix_type, matrix_precision, N, M);
    bml_diagonalize(A, eigenvalues, eigenvectors);
    bml_print_dense_vector(N, double_real, eigenvalues, 0, N);
    bml_print_bml_matrix(eigenvectors, 0, N, 0, N);

    bml_deallocate(&A);
    bml_deallocate(&A_t);
    bml_deallocate(&eigenvectors);
    free(eigenvalues);

    LOG_INFO("diagonalize matrix test passed\n");

    return 0;
}
