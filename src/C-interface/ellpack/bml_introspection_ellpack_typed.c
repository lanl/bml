#include "../../macros.h"
#include "../../typed.h"
#include "../bml_introspection.h"
#include "../bml_types.h"
#include "bml_introspection_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

/** Return the sparsity of a matrix.
 *
 *  Note that the the sparsity of a matrix is defined
 *  as NumberOfZeroes/N*N where N is the matrix dimension.
 *  To density of matrix A will be defined as 1-sparsity(A)
 *
 * \ingroup introspection_group_C
 *
 * \param A The bml matrix.
 * \param threshold The threshold used to compute the sparsity.
 * \return The sparsity of A.
 */
double TYPED_FUNC(
    bml_get_sparsity_ellpack) (
    bml_matrix_ellpack_t * A,
    double threshold)
{
    int nnzs = 0;
    double sparsity;
    REAL_T *A_value = (REAL_T *) A->value;
    int A_N = A->N;
    int A_M = A->M;
    int *A_nnz = A->nnz;

#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(A_value[:A_N*A_M], A_nnz[:A_N])
#endif

    for (int i = 0; i < A->N; i++)
    {
        for (int j = 0; j < A_nnz[i]; j++)
        {
            if (ABS(A_value[ROWMAJOR(i, j, A_N, A_M)]) > threshold)
            {
                nnzs++;
            }
        }
    }

    sparsity = (1.0 - (double) nnzs / ((double) (A_N * A_N)));
    return sparsity;
}
