#ifdef BML_USE_MAGMA
//define boolean data type needed by magma
#include <stdbool.h>
#include "magma_v2.h"
#endif

#include "../../macros.h"
#include "../../typed.h"
#include "../blas.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_trace.h"
#include "../bml_types.h"
#include "bml_trace_dense.h"
#include "bml_types_dense.h"
#include "bml_allocate_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Calculate the trace of a matrix.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix to calculate a trace for
 *  \return The trace of A
 */
double TYPED_FUNC(
    bml_trace_dense) (
    bml_matrix_dense_t * A)
{
    int N = A->N;

    REAL_T trace = 0.0;
#ifdef BML_USE_MAGMA
    MAGMA_T *dtmp;
    int ld = magma_roundup(N, 32);
    magma_int_t ret = MAGMA(malloc) ((MAGMA_T **) & dtmp, ld);
    assert(ret == MAGMA_SUCCESS);

    MAGMA_T *htmp = malloc(N * sizeof(MAGMA_T));
    for (int i = 0; i < N; i++)
        htmp[i] = MAGMACOMPLEX(MAKE) (1., 0.);

    MAGMA(setvector) (N, htmp, 1, dtmp, 1, bml_queue());
#if defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
    MAGMA_T ttrace = MAGMA(dotu) (N, (MAGMA_T *) A->matrix, A->ld + 1,
                                  dtmp, 1, bml_queue());
    trace = MAGMACOMPLEX(REAL) (ttrace) + I * MAGMACOMPLEX(IMAG) (ttrace);
#else
    trace = MAGMA(dot) (N, (REAL_T *) A->matrix, A->ld + 1,
                        dtmp, 1, bml_queue());
#endif

    magma_free(dtmp);
    free(htmp);
#else
    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    REAL_T *A_matrix = A->matrix;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];

#ifdef MKL_GPU
#pragma omp target update from(A_matrix[0:N*N])
#endif
#pragma omp parallel for                        \
  shared(N, A_matrix)                           \
  shared(rowMin, rowMax, myRank)  \
  reduction(+:trace)
    //for (int i = 0; i < N; i++)
    for (int i = rowMin; i < rowMax; i++)
    {
        trace += A_matrix[ROWMAJOR(i, i, N, N)];
    }
#endif
    return (double) REAL_PART(trace);
}

/** Calculate the trace of a matrix multiplication. The matrices must
 * be of the same size and symmetric.
 *
 *  \ingroup trace_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \return The trace of A*B
 */
double TYPED_FUNC(
    bml_trace_mult_dense) (
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B)
{
    int N = A->N;

    REAL_T trace = 0.0;

    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];

    if (N != B->N)
    {
        LOG_ERROR
            ("bml_trace_mult_dense: Matrices A and B are different sizes.");
    }

#ifdef BML_USE_MAGMA

    for (int i = A_localRowMin[myRank]; i < A_localRowMax[myRank]; i++)
    {
#if defined(SINGLE_COMPLEX) || defined(DOUBLE_COMPLEX)
        MAGMA_T ttrace = MAGMA(dotu) (N, (MAGMA_T *) A->matrix + i * A->ld, 1,
                                      (MAGMA_T *) B->matrix + i * B->ld, 1,
                                      bml_queue());
        trace +=
            MAGMACOMPLEX(REAL) (ttrace) + I * MAGMACOMPLEX(IMAG) (ttrace);
#else
        trace += MAGMA(dot) (N, (REAL_T *) A->matrix + i * A->ld, 1,
                             (REAL_T *) B->matrix + i * B->ld, 1,
                             bml_queue());
#endif
    }

#else

    REAL_T *A_matrix = A->matrix;
    REAL_T *B_matrix = B->matrix;

#ifdef MKL_GPU
#pragma omp target data map(A_matrix[0:N*N], B_matrix[0:N*N])
#endif
#pragma omp parallel for                        \
  shared(N, A_matrix, B_matrix)                 \
  shared(rowMin, rowMax, myRank)  \
  reduction(+:trace)
    for (int i = rowMin * N; i < rowMax * N; i++)
    {
        trace += A_matrix[i] * B_matrix[i];
    }

#ifdef MKL_GPU
//#pragma omp target update from(trace[0:1])
#endif
#endif

    return (double) REAL_PART(trace);
}
