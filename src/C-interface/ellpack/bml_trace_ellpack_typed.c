#include "../../macros.h"
#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_submatrix.h"
#include "../bml_trace.h"
#include "../bml_types.h"
#include "bml_submatrix_ellpack.h"
#include "bml_trace_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Calculate the trace of a matrix.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix to calculate a trace for
 *  \return the trace of A
 */
double TYPED_FUNC(
    bml_trace_ellpack) (
    bml_matrix_ellpack_t * A)
{
    int N = A->N;
    int M = A->M;

    int *A_index = (int *) A->index;
    int *A_nnz = (int *) A->nnz;
    int *A_localRowMin = (int *) A->domain->localRowMin;
    int *A_localRowMax = (int *) A->domain->localRowMax;

    REAL_T trace = 0.0;
    REAL_T *A_value = (REAL_T *) A->value;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];
    int numrows = rowMax - rowMin;

#ifdef USE_OMP_OFFLOAD
    REAL_T *diag;
    diag = (REAL_T *) calloc(numrows, sizeof(REAL_T));
#pragma omp target enter data map(to:diag[:numrows])
#pragma omp target teams distribute
    //for (int i = 0; i < N; i++)
    for (int i = rowMin; i < rowMax; i++)
    {
        REAL_T this_trace = 0.0;
#pragma omp parallel for shared(this_trace)
        for (int j = 0; j < A_nnz[i]; j++)
        {
            if (i == A_index[ROWMAJOR(i, j, N, M)])
            {
                this_trace = A_value[ROWMAJOR(i, j, N, M)];
            }
        }
        diag[i - rowMin] = this_trace;
    }
#pragma omp target teams distribute parallel for reduction(+:trace)
    for (int i = 0; i < numrows; i++)
        trace += diag[i];
#pragma omp target exit data map(delete:diag[:numrows])
    free(diag);
#else
#pragma omp parallel for                        \
  shared(A_value, A_index, A_nnz)         \
  reduction(+:trace)
    //for (int i = 0; i < N; i++)
    for (int i = rowMin; i < rowMax; i++)
    {
        for (int j = 0; j < A_nnz[i]; j++)
        {
            if (i == A_index[ROWMAJOR(i, j, N, M)])
            {
                trace += A_value[ROWMAJOR(i, j, N, M)];
                break;
            }
        }
    }
#endif

    return (double) REAL_PART(trace);
}

/** Calculate the trace of a matrix multiplication.
 * Both matrices must have the same size.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix A
 *  \param A The matrix B
 *  \return the trace of A*B
 */
double TYPED_FUNC(
    bml_trace_mult_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B)
{
    int A_N = A->N;
    int A_M = A->M;

    int *A_index = (int *) A->index;
    int *A_nnz = (int *) A->nnz;
    int *A_localRowMin = (int *) A->domain->localRowMin;
    int *A_localRowMax = (int *) A->domain->localRowMax;

    REAL_T trace = 0.0;
    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *rvalue;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];

    if (A_N != B->N || A_M != B->M)
    {
        LOG_ERROR
            ("bml_trace_mult_ellpack: Matrices A and B have different sizes.");
    }

    int B_N = B->N;
    int B_M = B->M;

    REAL_T *B_value = (REAL_T *) B->value;
    int *B_index = (int *) B->index;
    int *B_nnz = (int *) B->nnz;
#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(A_nnz[:A_N], A_index[:A_N*A_M], A_value[:A_N*A_M])
#pragma omp target update from(B_nnz[:B_N], B_index[:B_N*B_M], B_value[:B_N*B_M])
#endif

#pragma omp parallel for                        \
  private(rvalue)                               \
  shared(B, A_N, A_M, A_value, A_index, A_nnz)  \
  shared(A_localRowMin, A_localRowMax, myRank)  \
  reduction(+:trace)
    //for (int i = 0; i < A_N; i++)
    for (int i = rowMin; i < rowMax; i++)
    {
        rvalue =
            TYPED_FUNC(bml_getVector_ellpack) (B,
                                               &A_index[ROWMAJOR
                                                        (i, 0, A_N, A_M)], i,
                                               A_nnz[i]);
        for (int j = 0; j < A_nnz[i]; j++)
        {
            trace += A_value[ROWMAJOR(i, j, A_N, A_M)] * rvalue[j];
        }
        bml_free_memory(rvalue);
    }

    return (double) REAL_PART(trace);
}
