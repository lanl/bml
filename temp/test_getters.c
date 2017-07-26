#include "macros.h"

#include <bml.h>
#include <math.h>
#include <stdlib.h>

#define REL_TOL 1e-12
#define N 11
#define M 11

int main(int argc, char **argv)
{
    bml_matrix_t *A = NULL;
    double *A_dense = NULL;

    A_dense = calloc(N * N, sizeof(double));

    for (int i = 0; i < N * N; i++)
    {
        A_dense[i] = (double) (rand() / (double) RAND_MAX);
    }

    A = bml_import_from_dense(dense, double_real, dense_row_major,
                              N, M, A_dense, 0.0, sequential);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double *Aij = bml_get(A, i, j);
            double expected = A_dense[ROWMAJOR(i, j, N, N)];
            double rel_diff = fabs((expected - *Aij) / expected);
            if (rel_diff > REL_TOL)
            {
                LOG_ERROR
                    ("matrices are not identical; expected[%d] = %e, B[%d] = %e\n",
                     i, expected, i, *Aij);
                return -1;
            }
        }
    }

    LOG_INFO("test passed\n");
}
