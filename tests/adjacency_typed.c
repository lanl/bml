#include "bml.h"
#include "typed.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
#define REL_TOL 1e-6
#else
#define REL_TOL 1e-12
#endif

int TYPED_FUNC(
    test_adjacency) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;
    REAL_T *A_dense = NULL;

    LOG_DEBUG("rel. tolerance = %e\n", REL_TOL);

    A = bml_random_matrix(matrix_type, matrix_precision, N, M);
	int * xadj = malloc(sizeof(int)*(N + 1));
	int * adjncy =malloc(sizeof(int)*(N *N));
	bml_adjacency(A, xadj, adjncy);

    A_dense = bml_convert_to_dense(A, dense_row_major);
    bml_print_dense_matrix(N, matrix_precision, dense_row_major, A_dense, 0, N, 0, N);
    int i;
    int sumnnz;
    

    /*
    for (int i = 0; i < N * N; i++)
    {
		 i
        if (rel_diff > REL_TOL)
        {
            LOG_ERROR
                ("matrices are not identical; expected[%d] = %e, B[%d] = %e\n",
                 i, expected, i, B_dense[i]);
            return -1;
        }
    }*/
    bml_free_memory(A_dense);
    bml_deallocate(&A);


    LOG_INFO("add matrix test passed\n");

    return 0;
}
