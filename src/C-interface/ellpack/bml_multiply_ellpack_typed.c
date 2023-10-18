#include "../../macros.h"
#include "../../typed.h"
#include "../bml_add.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_multiply.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_add_ellpack.h"
#include "bml_allocate_ellpack.h"
#include "bml_multiply_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef BML_USE_CUSPARSE
#include "bml_trace_ellpack.h"
#include <cusparse.h>
#endif
#ifdef BML_USE_ROCSPARSE
#include "bml_trace_ellpack.h"
// Copy rocsparse headers into src/C-insterface/rocsparse/ and edit rocsparse_functions.h to remove '[[...]]' text
#include "../rocsparse/rocsparse.h"
//#include <hip/hip_runtime.h> // needed for hipDeviceSynchronize()
#endif

#if defined(BML_USE_HYPRE)
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "seq_mv.h"
#endif

/** Matrix multiply.
 *
 * \f$ C \leftarrow \alpha A \, B + \beta C \f$
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param alpha Scalar factor multiplied by A * B
 * \param beta Scalar factor multiplied by C
 * \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_multiply_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha,
    double beta,
    double threshold)
{
    double ONE = 1.0;
    double ZERO = 0.0;

    void *trace = NULL;

    if (A == NULL || B == NULL)
    {
        LOG_ERROR("Either matrix A or B are NULL\n");
    }
// To Do: Direct implementation of (C = alpha*A*B + beta*C) to reduce data motion -DOK
//#if defined(BML_USE_CUSPARSE)
//    TYPED_FUNC(bml_multiply_cusparse_ellpack) (A, B, C, alpha, beta,
//                                               threshold);
#if defined(BML_USE_ROCSPARSE)
    TYPED_FUNC(bml_multiply_rocsparse_ellpack) (A, B, C, alpha, beta,
                                                threshold);
#elif defined(BML_USE_HYPRE)
    TYPED_FUNC(bml_multiply_hypre_ellpack) (A, B, C, alpha, beta,
                                                threshold);
  
#else
    if (A == B && alpha == ONE && beta == ZERO)
    {
        trace = TYPED_FUNC(bml_multiply_x2_ellpack) (A, C, threshold);
    }
    else
    {
        bml_matrix_dimension_t matrix_dimension = { C->N, C->N, C->M };
        bml_matrix_ellpack_t *A2 =
            TYPED_FUNC(bml_noinit_matrix_ellpack) (matrix_dimension,
                                                   A->distribution_mode);

        if (A != NULL && A == B)
        {
            trace = TYPED_FUNC(bml_multiply_x2_ellpack) (A, A2, threshold);
        }
        else
        {
            TYPED_FUNC(bml_multiply_AB_ellpack) (A, B, A2, threshold);
        }

#ifdef BML_USE_MPI
        if (bml_getNRanks() > 1 && A2->distribution_mode == distributed)
        {
            bml_allGatherVParallel(A2);
        }
#endif

        TYPED_FUNC(bml_add_ellpack) (C, A2, beta, alpha, threshold);

        bml_deallocate_ellpack(A2);
    }
#endif
    bml_free_memory(trace);
}

/** Matrix multiply.
 *
 * \f$ X^{2} \leftarrow X \, X \f$
 *
 * \ingroup multiply_group
 *
 * \param X Matrix X
 * \param X2 Matrix X2
 * \param threshold Used for sparse multiply
 */
void *TYPED_FUNC(
    bml_multiply_x2_ellpack) (
    bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    double threshold)
{
    int *X_localRowMin = X->domain->localRowMin;
    int *X_localRowMax = X->domain->localRowMax;

    int X_N = X->N;
    int X_M = X->M;
    int *X_index = X->index;
    int *X_nnz = X->nnz;

    int X2_N = X2->N;
    int X2_M = X2->M;
    int *X2_index = X2->index;
    int *X2_nnz = X2->nnz;

    REAL_T traceX = 0.0;
    REAL_T traceX2 = 0.0;
    REAL_T *X_value = (REAL_T *) X->value;
    REAL_T *X2_value = (REAL_T *) X2->value;

    double *trace = bml_allocate_memory(sizeof(double) * 2);

    int myRank = bml_getMyRank();
    int rowMin = X_localRowMin[myRank];
    int rowMax = X_localRowMax[myRank];

#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE) || defined(BML_USE_HYPRE)
    double alpha = 1.0;
    double beta = 0.0;

#if defined(BML_USE_CUSPARSE)
    /* cusparse seems to accept passing the same matrix for A and B in the spGEMM api */
    TYPED_FUNC(bml_multiply_cusparse_ellpack) (X, X, X2, alpha, beta,
                                               threshold);
#elif defined(BML_USE_ROCSPARSE)

    TYPED_FUNC(bml_multiply_rocsparse_ellpack) (X, X, X2, alpha, beta,
                                                threshold);
#elif defined(BML_USE_HYPRE)
    TYPED_FUNC(bml_multiply_hypre_ellpack) (X, X, X2, alpha, beta,
                                               threshold);
#endif

    traceX = TYPED_FUNC(bml_trace_ellpack) (X);
    traceX2 = TYPED_FUNC(bml_trace_ellpack) (X2);

#else

    //Should be safe to use BML_OFFLOAD_CHUNKS here but preserving old version
    //#if !(defined(__IBMC__) || defined(__ibmxl__) || (defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK))))
#if !(defined(__IBMC__) || defined(__ibmxl__) || (defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS ))
    int ix[X_N], jx[X_N];
    REAL_T x[X_N];

    memset(ix, 0, X_N * sizeof(int));
    memset(jx, 0, X_N * sizeof(int));
    memset(x, 0.0, X_N * sizeof(REAL_T));
#endif

#if defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS
    int num_chunks = MIN(BML_OFFLOAD_NUM_CHUNKS, rowMax - rowMin + 1);

    int *all_ix, *all_jx;
    REAL_T *all_x;

    all_ix = calloc(X_N * num_chunks, sizeof(int));
    all_jx = calloc(X_N * num_chunks, sizeof(int));
    all_x = calloc(X_N * num_chunks, sizeof(REAL_T));

#pragma omp target enter data map(to:all_ix[0:X_N*num_chunks],all_jx[0:X_N*num_chunks],all_x[0:X_N*num_chunks])

#endif

#if defined (USE_OMP_OFFLOAD)
#pragma omp target
#if BML_OFFLOAD_CHUNKS
#pragma omp teams distribute parallel for	\
    shared(X_N, X_M, X_index, X_nnz, X_value)  \
    shared(X2_N, X2_M, X2_index, X2_nnz, X2_value)     \
    shared(rowMin, rowMax)                             \
    reduction(+: traceX, traceX2)
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int *ix, *jx;
        REAL_T *x;

        ix = &all_ix[chunk * X_N];
        jx = &all_jx[chunk * X_N];
        x = &all_x[chunk * X_N];

#else
#pragma omp teams distribute parallel for                               \
    shared(X_N, X_M, X_index, X_nnz, X_value)  \
    shared(X2_N, X2_M, X2_index, X2_nnz, X2_value)     \
    shared(rowMin, rowMax)                             \
    firstprivate(ix,jx, x)                             \
    reduction(+: traceX, traceX2)
#endif
#else
#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for                               \
    shared(X_N, X_M, X_index, X_nnz, X_value)  \
    shared(X2_N, X2_M, X2_index, X2_nnz, X2_value)     \
    shared(rowMin, rowMax)                             \
    reduction(+: traceX, traceX2)
#else
#if !(defined(CRAY_SDK) || defined(INTEL_SDK))
#pragma vector aligned
#endif
#pragma omp parallel for                               \
    shared(X_N, X_M, X_index, X_nnz, X_value)  \
    shared(X2_N, X2_M, X2_index, X2_nnz, X2_value)     \
    shared(rowMin, rowMax)                             \
    firstprivate(ix,jx, x)                             \
    reduction(+: traceX, traceX2)
#endif
#endif

#if defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS
    for (int i = rowMin + chunk; i < rowMax; i = i + num_chunks)
    {
#else
    for (int i = rowMin; i < rowMax; i++)
    {

#if defined(__IBMC__) || defined(__ibmxl__)
        int ix[X_N], jx[X_N];
        REAL_T x[X_N];

        memset(ix, 0, X_N * sizeof(int));
#endif
#endif
#ifdef INTEL_OPT
        __assume_aligned(X_nnz, MALLOC_ALIGNMENT);
        __assume_aligned(X_index, MALLOC_ALIGNMENT);
        __assume_aligned(X_value, MALLOC_ALIGNMENT);
#endif
        int l = 0;
        for (int jp = 0; jp < X_nnz[i]; jp++)
        {
            REAL_T a = X_value[ROWMAJOR(i, jp, X_N, X_M)];
            int j = X_index[ROWMAJOR(i, jp, X_N, X_M)];
            if (j == i)
            {
                traceX = traceX + a;
            }
            for (int kp = 0; kp < X_nnz[j]; kp++)
            {
                int k = X_index[ROWMAJOR(j, kp, X_N, X_M)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    //X2_index[ROWMAJOR(i, l, N, M)] = k;
                    jx[l] = k;
                    ix[k] = i + 1;
                    l++;
                }
                // TEMPORARY STORAGE VECTOR LENGTH FULL N
                x[k] = x[k] + a * X_value[ROWMAJOR(j, kp, X_N, X_M)];
            }
        }

#ifdef INTEL_OPT
        __assume_aligned(X2_nnz, MALLOC_ALIGNMENT);
        __assume_aligned(X2_index, MALLOC_ALIGNMENT);
        __assume_aligned(X2_value, MALLOC_ALIGNMENT);
#endif
        int ll = 0;
        for (int j = 0; j < l; j++)
        {
            int jp = jx[j];
            REAL_T xtmp = x[jp];
            if (jp == i)
            {
                traceX2 = traceX2 + xtmp;
                X2_value[ROWMAJOR(i, ll, X2_N, X2_M)] = xtmp;
                X2_index[ROWMAJOR(i, ll, X2_N, X2_M)] = jp;
                ll++;
            }
            else if (is_above_threshold(xtmp, threshold))
            {
                X2_value[ROWMAJOR(i, ll, X2_N, X2_M)] = xtmp;
                X2_index[ROWMAJOR(i, ll, X2_N, X2_M)] = jp;
                ll++;
            }
            ix[jp] = 0;
            x[jp] = 0.0;
        }
        // Check for number of non-zeroes per row exceeded
        // We should do it before assigning values to X2_value above,
        // but that would be a lot of checks
        if (ll > X2_M)
        {
#ifndef USE_OMP_OFFLOAD
            LOG_ERROR("Number of non-zeroes per row > M, Increase M\n");
#endif
        }

        X2_nnz[i] = ll;
    }

#if defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS
}
#pragma omp target exit data map(delete:all_ix[0:X_N*num_chunks],all_jx[0:X_N*num_chunks],all_x[0:X_N*num_chunks])
    free(all_ix);
    free(all_jx);
    free(all_x);
#endif

#endif // endif cusparse or rocsparse

trace[0] = traceX;
trace[1] = traceX2;

return trace;
}

/** Matrix multiply.
 *
 * \f$ C \leftarrow B \, A \f$
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_multiply_AB_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold)
{
    int A_N = A->N;
    int A_M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    int B_N = B->N;
    int B_M = B->M;
    int *B_nnz = B->nnz;
    int *B_index = B->index;

    int C_N = C->N;
    int C_M = C->M;
    int *C_nnz = C->nnz;
    int *C_index = C->index;

    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;
    REAL_T *C_value = (REAL_T *) C->value;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];

#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE) || defined(BML_USE_HYPRE)
    double alpha = 1.0;
    double beta = 0.0;

#if defined(BML_USE_CUSPARSE)
    TYPED_FUNC(bml_multiply_cusparse_ellpack) (A, B, C, alpha, beta,
                                               threshold);
#elif defined(BML_USE_ROCSPARSE)
    TYPED_FUNC(bml_multiply_rocsparse_ellpack) (A, B, C, alpha, beta,
                                                threshold);
#elif defined(BML_USE_HYPRE)
    TYPED_FUNC(bml_multiply_hypre_ellpack) (A, B, C, alpha, beta,
                                               threshold);
#endif

#else

    //Should be safe to use BML_OFFLOAD_CHUNKS here but preserving old version
    //#if !(defined(__IBMC__) || defined(__ibmxl__) || (defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK))))
#if !(defined(__IBMC__) || defined(__ibmxl__) || (defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS ))
    int ix[C->N], jx[C->N];
    REAL_T x[C->N];

    memset(ix, 0, C->N * sizeof(int));
    memset(jx, 0, C->N * sizeof(int));
    memset(x, 0.0, C->N * sizeof(REAL_T));
#endif

#if defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS
    int num_chunks = MIN(BML_OFFLOAD_NUM_CHUNKS, rowMax - rowMin + 1);

    int *all_ix, *all_jx;
    REAL_T *all_x;

    all_ix = calloc(C_N * num_chunks, sizeof(int));
    all_jx = calloc(C_N * num_chunks, sizeof(int));
    all_x = calloc(C_N * num_chunks, sizeof(REAL_T));

#pragma omp target enter data map(to:all_ix[0:C_N*num_chunks],all_jx[0:C_N*num_chunks],all_x[0:C_N*num_chunks])

#endif

#if defined (USE_OMP_OFFLOAD)
#pragma omp target
#if BML_OFFLOAD_CHUNKS
#pragma omp teams distribute parallel for \
    shared(A_N, A_M, A_nnz, A_index, A_value)  \
    shared(A_localRowMin, A_localRowMax)       \
    shared(B_N, B_M, B_nnz, B_index, B_value)  \
    shared(C_N, C_M, C_nnz, C_index, C_value)
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int *ix, *jx;
        REAL_T *x;

        ix = &all_ix[chunk * C_N];
        jx = &all_jx[chunk * C_N];
        x = &all_x[chunk * C_N];

#else
#pragma omp teams distribute parallel for \
    shared(A_N, A_M, A_nnz, A_index, A_value)  \
    shared(A_localRowMin, A_localRowMax)       \
    shared(B_N, B_M, B_nnz, B_index, B_value)  \
    shared(C_N, C_M, C_nnz, C_index, C_value)  \
    firstprivate(ix, jx, x)
#endif
#else
#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for                       \
    shared(A_N, A_M, A_nnz, A_index, A_value)  \
    shared(A_localRowMin, A_localRowMax)       \
    shared(B_N, B_M, B_nnz, B_index, B_value)  \
    shared(C_N, C_M, C_nnz, C_index, C_value)
#else
#pragma omp parallel for                       \
    shared(A_N, A_M, A_nnz, A_index, A_value)  \
    shared(A_localRowMin, A_localRowMax)       \
    shared(B_N, B_M, B_nnz, B_index, B_value)  \
    shared(C_N, C_M, C_nnz, C_index, C_value)  \
    firstprivate(ix, jx, x)
#endif
#endif
    //for (int i = 0; i < A_N; i++)
#if defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS
    for (int i = rowMin + chunk; i < rowMax; i = i + num_chunks)
    {
#else
    for (int i = rowMin; i < rowMax; i++)
    {
#if defined(__IBMC__) || defined(__ibmxl__)
        int ix[C_N], jx[C_N];
        REAL_T x[C_N];

        memset(ix, 0, C_N * sizeof(int));
#endif
#endif
        int l = 0;
        for (int jp = 0; jp < A_nnz[i]; jp++)
        {
            REAL_T a = A_value[ROWMAJOR(i, jp, A_N, A_M)];
            int j = A_index[ROWMAJOR(i, jp, A_N, A_M)];

            for (int kp = 0; kp < B_nnz[j]; kp++)
            {
                int k = B_index[ROWMAJOR(j, kp, B_N, B_M)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    //C_index[ROWMAJOR(i, l, N, M)] = k;
                    jx[l] = k;
                    ix[k] = i + 1;
                    l++;
                }
                // TEMPORARY STORAGE VECTOR LENGTH FULL N
                x[k] = x[k] + a * B_value[ROWMAJOR(j, kp, B_N, B_M)];
            }
        }

        // Check for number of non-zeroes per row exceeded
        if (l > C_M)
        {
#ifndef USE_OMP_OFFLOAD
            LOG_ERROR("Number of non-zeroes per row > M, Increase M\n");
#endif
        }

        int ll = 0;
        for (int j = 0; j < l; j++)
        {
            //int jp = C_index[ROWMAJOR(i, j, N, M)];
            int jp = jx[j];
            REAL_T xtmp = x[jp];
            if (jp == i)
            {
                C_value[ROWMAJOR(i, ll, C_N, C_M)] = xtmp;
                C_index[ROWMAJOR(i, ll, C_N, C_M)] = jp;
                ll++;
            }
            else if (is_above_threshold(xtmp, threshold))
            {
                C_value[ROWMAJOR(i, ll, C_N, C_M)] = xtmp;
                C_index[ROWMAJOR(i, ll, C_N, C_M)] = jp;
                ll++;
            }
            ix[jp] = 0;
            x[jp] = 0.0;
        }
        C_nnz[i] = ll;
    }
#if defined(USE_OMP_OFFLOAD) && BML_OFFLOAD_CHUNKS
}
#pragma omp target exit data map(delete:all_ix[0:C_N*num_chunks],all_jx[0:C_N*num_chunks],all_x[0:C_N*num_chunks])
    free(all_ix);
    free(all_jx);
    free(all_x);
#endif

#endif // endif cusparse or rocsparse
}

/** Matrix multiply with threshold adjustment.
 *
 * \f$ C \leftarrow B \, A \f$
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_multiply_adjust_AB_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold)
{
    int A_N = A->N;
    int A_M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    int B_N = B->N;
    int B_M = B->M;
    int *B_nnz = B->nnz;
    int *B_index = B->index;

    int C_N = C->N;
    int C_M = C->M;
    int *C_nnz = C->nnz;
    int *C_index = C->index;

    int aflag = 1;

    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;
    REAL_T *C_value = (REAL_T *) C->value;

    REAL_T adjust_threshold = (REAL_T) threshold;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];

#ifdef USE_OMP_OFFLOAD
#pragma omp target update from(A_nnz[:A_N], A_index[:A_N*A_M], A_value[:A_N*A_M])
#pragma omp target update from(B_nnz[:B_N], B_index[:B_N*B_M], B_value[:B_N*B_M])
#endif

#if !(defined(__IBMC__) || defined(__ibmxl__))
    int ix[C->N], jx[C->N];
    REAL_T x[C->N];

    memset(ix, 0, C->N * sizeof(int));
    memset(jx, 0, C->N * sizeof(int));
    memset(x, 0.0, C->N * sizeof(REAL_T));
#endif

    while (aflag > 0)
    {
        aflag = 0;

#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for                       \
    shared(A_N, A_M, A_nnz, A_index, A_value)  \
    shared(B_N, B_M, B_nnz, B_index, B_value)  \
    shared(C_N, C_M, C_nnz, C_index, C_value)  \
    shared(adjust_threshold)           \
    reduction(+:aflag)
#else
#pragma omp parallel for                       \
    shared(A_N, A_M, A_nnz, A_index, A_value)  \
    shared(B_N, B_M, B_nnz, B_index, B_value)  \
    shared(C_N, C_M, C_nnz, C_index, C_value)  \
    shared(adjust_threshold)           \
    firstprivate(ix, jx, x)                    \
    reduction(+:aflag)
#endif

        //for (int i = 0; i < A_N; i++)
        for (int i = rowMin; i < rowMax; i++)
        {

#if defined(__IBMC__) || defined(__ibmxl__)
            int ix[C_N], jx[C_N];
            REAL_T x[C_N];

            memset(ix, 0, C_N * sizeof(int));
#endif

            int l = 0;
            for (int jp = 0; jp < A_nnz[i]; jp++)
            {
                REAL_T a = A_value[ROWMAJOR(i, jp, A_N, A_M)];
                int j = A_index[ROWMAJOR(i, jp, A_N, A_M)];

                for (int kp = 0; kp < B_nnz[j]; kp++)
                {
                    int k = B_index[ROWMAJOR(j, kp, B_N, B_M)];
                    if (ix[k] == 0)
                    {
                        x[k] = 0.0;
                        jx[l] = k;
                        ix[k] = i + 1;
                        l++;
                    }
                    // TEMPORARY STORAGE VECTOR LENGTH FULL N
                    x[k] = x[k] + a * B_value[ROWMAJOR(j, kp, B_N, B_M)];
                }
            }

            // Check for number of non-zeroes per row exceeded
            // Need to adjust threshold
            // ** Shouldn't we break from this loop if aflag is 1?? --DOK
            if (l > C_M)
            {
                aflag = 1;
            }

            int ll = 0;
            for (int j = 0; j < l; j++)
            {
                //int jp = C_index[ROWMAJOR(i, j, N, M)];
                int jp = jx[j];
                REAL_T xtmp = x[jp];
                // Diagonal elements are saved in first column
                if (jp == i)
                {
                    C_value[ROWMAJOR(i, ll, C_N, C_M)] = xtmp;
                    C_index[ROWMAJOR(i, ll, C_N, C_M)] = jp;
                    ll++;
                }
                else if (is_above_threshold(xtmp, adjust_threshold))
                {
                    C_value[ROWMAJOR(i, ll, C_N, C_M)] = xtmp;
                    C_index[ROWMAJOR(i, ll, C_N, C_M)] = jp;
                    ll++;
                }
                ix[jp] = 0;
                x[jp] = 0.0;
            }
            C_nnz[i] = ll;
        }

        adjust_threshold *= (REAL_T) 2.0;
    }
#ifdef USE_OMP_OFFLOAD
#pragma omp target update to(C_nnz[:C_N], C_index[:C_N*C_M], C_value[:C_N*C_M])
#endif
}


#if defined(BML_USE_CUSPARSE)
/** cuSPARSE matrix multiply.
 * Note: This function assumes that C is allocated and has enough memory to hold the result.
 * While this is not required by cusparse, BML generally allocates the matrices passed in as arguments.
 * This function exploits that to avoid allocating temporary data to hold the result (and copying back
 * to C), by directly placing the result in C. Thus the result is undefined if the number of nonzeros
 * in the rows of the resulting C matrix is larger than the M dimension of C. We do not check that
 * these sizes are consistent.
 *
 * \f$ C \leftarrow \alpha A \, B + \beta C \f$
 *
 * NOTE!! cusparse's documentation suggests that the routines used here should be able to perform the
 * above operation (C = alpha*A*B + beta*C). However, it appears that this assumes the input C and the
 * result of A*B have the same sparsity pattern. Otherwise, the sparsity pattern of A*B takes precedence.
 * For now, we assume beta = 0. to ensure that we get C = alpha*A*B. This routine should be updated later
 * to internally do the addition when the data is already in the cusparse arrays.
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_multiply_cusparse_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha1,
    double beta1,
    double threshold1)
{
    int A_N = A->N;
    int A_M = A->M;

    int B_N = B->N;
    int B_M = B->M;

    int C_N = C->N;
    int C_M = C->M;

    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;
    REAL_T *C_value = (REAL_T *) C->value;

    int *csrColIndA = A->csrColInd;
    int *csrColIndB = B->csrColInd;
    int *csrColIndC = C->csrColInd;
    int *csrColIndC_tmp = NULL;
    int *csrRowPtrA = A->csrRowPtr;
    int *csrRowPtrB = B->csrRowPtr;
    int *csrRowPtrC = C->csrRowPtr;
    int *csrRowPtrC_tmp = NULL;
    REAL_T *csrValA = (REAL_T *) A->csrVal;
    REAL_T *csrValB = (REAL_T *) B->csrVal;
    REAL_T *csrValC = (REAL_T *) C->csrVal;
    REAL_T *csrValC_tmp = NULL;

    /* temporary arrays to hold initial C values */
    int *d_ccols = NULL;
    int *d_rptr = NULL;
    REAL_T *d_cvals = NULL;

    REAL_T alpha = (REAL_T) alpha1;
    REAL_T beta = (REAL_T) beta1;
    // force beta = 0. (See Note!! above) -DOK
    beta = 0.;
    REAL_T threshold = (REAL_T) threshold1;

    // Variables used for pruning the final matrix
    int nnzC = 0;
    size_t lworkInBytes = 0;
    char *dwork = NULL;

    // cuSPARSE APIs
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cudaDataType computeType = BML_CUSPARSE_T;

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC, matC_tmp;

    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    // convert ellpack to cucsr
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (A);
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (B);
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (C);

    // Create cusparse matrix A and B in CSR format
    // Note: The following update is not necessary since the ellpack2cucsr
    // routine updates the csr rowpointers on host and device
#pragma omp target update from(csrRowPtrA[:A_N+1])
#pragma omp target update from(csrRowPtrB[:B_N+1])
#pragma omp target update from(csrRowPtrC[:C_N+1])
    int nnzA = csrRowPtrA[A_N];
    int nnzB = csrRowPtrB[B_N];
    int nnzC_in = csrRowPtrC[C_N];

    // copy C into temporary arrays if nonzero
/*
    if(nnzC_in > 0)
    {
        // allocate memory
        d_rptr = (int *) omp_target_alloc((C_N+1)*sizeof(int), omp_get_default_device());
        d_ccols = (int *) omp_target_alloc(nnzC_in*sizeof(int), omp_get_default_device());
        d_cvals = (REAL_T *) omp_target_alloc(nnzC_in*sizeof(REAL_T), omp_get_default_device());
        // copy
        omp_target_memcpy(d_rptr, csrRowPtrC,
                                  (C_N + 1) * sizeof(int), 0, 0,
                                  omp_get_default_device(),
                                  omp_get_default_device());
        omp_target_memcpy(d_ccols, csrColIndC,
                                  nnzC_in * sizeof(int), 0, 0,
                                  omp_get_default_device(),
                                  omp_get_default_device());
        omp_target_memcpy(d_cvals, csrValC,
                                  nnzC_in * sizeof(REAL_T), 0, 0,
                                  omp_get_default_device(),
                                  omp_get_default_device());
    }
*/
    // special case not well handled by cusparse

    if (nnzA == 0 || nnzB == 0)
    {
#pragma omp target teams distribute parallel for \
    shared(nnzC_in, csrValC)
        for (int i = 0; i < nnzC_in; i++)
        {
            csrValC[i] *= beta;
        }
        TYPED_FUNC(bml_cucsr2ellpack_ellpack) (C);
    }
    else
    {
        BML_CHECK_CUSPARSE(cusparseCreate(&handle));
#pragma omp target data use_device_ptr(csrRowPtrA,csrColIndA,csrValA, \
		csrRowPtrB,csrColIndB,csrValB)
        {
            BML_CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_N, A_N, nnzA,
                                                 csrRowPtrA, csrColIndA,
                                                 csrValA, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 computeType));
            BML_CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_N, B_N, nnzB,
                                                 csrRowPtrB, csrColIndB,
                                                 csrValB, CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 computeType));
            BML_CHECK_CUSPARSE(cusparseCreateCsr
                               (&matC, C_N, B_N, nnzC_in, csrRowPtrC,
                                csrColIndC, csrValC, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                computeType));
        }
        cusparseSpGEMMDescr_t spgemmDesc;
        BML_CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));
        // ask bufferSize1 bytes for external memory
        BML_CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                         &alpha, matA, matB,
                                                         &beta, matC,
                                                         computeType,
                                                         CUSPARSE_SPGEMM_DEFAULT,
                                                         spgemmDesc,
                                                         &bufferSize1, NULL));
        dBuffer1 =
            (void *) omp_target_alloc(bufferSize1, omp_get_default_device());
        // inspect the matrices A and B to understand the memory requirement fo\
        r
            // the next step
            BML_CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                             &alpha, matA,
                                                             matB, &beta,
                                                             matC,
                                                             computeType,
                                                             CUSPARSE_SPGEMM_DEFAULT,
                                                             spgemmDesc,
                                                             &bufferSize1,
                                                             dBuffer1));
        // ask bufferSize2 bytes for external memory
        BML_CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                                  &alpha, matA, matB, &beta,
                                                  matC, computeType,
                                                  CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDesc, &bufferSize2,
                                                  NULL));
        dBuffer2 =
            (void *) omp_target_alloc(bufferSize2, omp_get_default_device());

        // compute the intermediate product of A * B
        BML_CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                                  &alpha, matA, matB, &beta,
                                                  matC, computeType,
                                                  CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDesc, &bufferSize2,
                                                  dBuffer2));
        /* Note: Ideally we would allocate new arrays to hold the result and do a memcpy to the appropriate storage
         * after computing the product, since we do not know apriori the number of nonzeros of C. However, for ellpack,
         * we know the max nnz, so we can preallocate arrays of that size and use them here directly.
         */

#pragma omp target data use_device_ptr(csrRowPtrC,csrColIndC,csrValC)
        {
            // update matC with the new pointers (We use existing preallocated arrays here. See note above )
            BML_CHECK_CUSPARSE(cusparseCsrSetPointers
                               (matC, csrRowPtrC, csrColIndC, csrValC));
        }
        // copy the final products to the matrix C
        BML_CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB,
                                               &alpha, matA, matB, &beta,
                                               matC, computeType,
                                               CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc));
        // Done with matrix multiplication. Now prune to drop small entries. This is an "in-place" implementation
        // Note: cusparse has either cusparse<t>pruneCsr2csr or cusparse<t>csr2csr_compress to
        // accomplish this. We use cusparse<t>pruneCsr2csr here for convenience.
        // Prune allows the use of device pointers, whereas Compress works with managed memory.
        if (is_above_threshold(threshold, BML_REAL_MIN))
        {
            // Get matrix C data sizes ( required to be int64_t as of cusparse api v. 11)
            int64_t C_num_rows, C_num_cols, C_nnz_tmp;
            BML_CHECK_CUSPARSE(cusparseSpMatGetSize
                               (matC, &C_num_rows, &C_num_cols, &C_nnz_tmp));
            // create matrix descriptor for pruned matrix
            BML_CHECK_CUSPARSE(cusparseCreateMatDescr(&matC_tmp));
            BML_CHECK_CUSPARSE(cusparseSetMatIndexBase
                               (matC_tmp, CUSPARSE_INDEX_BASE_ZERO));
            BML_CHECK_CUSPARSE(cusparseSetMatType
                               (matC_tmp, CUSPARSE_MATRIX_TYPE_GENERAL));

            // Allocate temp array for holding row pointer data of pruned matrix.
            csrRowPtrC_tmp =
                (int *) omp_target_alloc(sizeof(int) * (C_N + 1),
                                         omp_get_default_device());

            /* Remaining work needs to use existing device pointers from C matrix so we create a target data region
             * to use these device pointers.
             */
#pragma omp target data use_device_ptr(csrRowPtrC,csrColIndC,csrValC)
            {
                // Get size of buffer and allocate
                BML_CHECK_CUSPARSE(bml_cusparsePruneCSR_bufferSizeExt
                                   (handle, C_num_rows, C_num_cols, C_nnz_tmp,
                                    (cusparseMatDescr_t) matC, csrValC,
                                    csrRowPtrC, csrColIndC, &threshold,
                                    matC_tmp, NULL, csrRowPtrC_tmp, NULL,
                                    &lworkInBytes));
                dwork =
                    (char *) omp_target_alloc(lworkInBytes,
                                              omp_get_default_device());

                // Estimate number of nonzeros of pruned matrix
                BML_CHECK_CUSPARSE(bml_cusparsePruneCSRNnz
                                   (handle, C_num_rows, C_num_cols, C_nnz_tmp,
                                    (cusparseMatDescr_t) matC, csrValC,
                                    csrRowPtrC, csrColIndC, &threshold,
                                    matC_tmp, csrRowPtrC_tmp,
                                    &nnzC /* host */ , dwork));
                // Prune the matrix into existing C matrix arrays (in-place)
                BML_CHECK_CUSPARSE(bml_cusparsePruneCSR
                                   (handle, C_num_rows, C_num_cols, C_nnz_tmp,
                                    (cusparseMatDescr_t) matC, csrValC,
                                    csrRowPtrC, csrColIndC, &threshold,
                                    matC_tmp, csrValC, csrRowPtrC_tmp,
                                    csrColIndC, dwork));
                // copy row pointer data from temp array to pruned result
                omp_target_memcpy(csrRowPtrC, csrRowPtrC_tmp,
                                  (C_N + 1) * sizeof(int), 0, 0,
                                  omp_get_default_device(),
                                  omp_get_default_device());
                // free work buffer
                omp_target_free(dwork, omp_get_default_device());
            }                   //end target data region

            // free tmp memory and matrix descriptor
            omp_target_free(csrRowPtrC_tmp, omp_get_default_device());
            BML_CHECK_CUSPARSE(cusparseDestroyMatDescr(matC_tmp));
        }

/*
// DEBUG:
#pragma omp target update from(csrRowPtrC[:N1])
#pragma omp target update from(csrValC[:C_nnz1])
#pragma omp target update from(csrColIndC[:C_nnz1])
for(int k=0; k<C_nnz1; k++)
{
   printf("%d, %f \n", csrColIndC[k], csrValC[k]);
}
*/

        // Done with matrix multiplication.
        // Update ellpack C matrix (on device): copy from csr to ellpack format
        TYPED_FUNC(bml_cucsr2ellpack_ellpack) (C);

        // device memory deallocation
        omp_target_free(dBuffer1, omp_get_default_device());
        omp_target_free(dBuffer2, omp_get_default_device());
        BML_CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
        BML_CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        BML_CHECK_CUSPARSE(cusparseDestroySpMat(matB));
        BML_CHECK_CUSPARSE(cusparseDestroySpMat(matC));
        BML_CHECK_CUSPARSE(cusparseDestroy(handle));
    }
}

#elif defined(BML_USE_ROCSPARSE)
void TYPED_FUNC(
    bml_multiply_rocsparse_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha1,
    double beta1,
    double threshold1)
{
    int A_N = A->N;
    int A_M = A->M;

    int B_N = B->N;
    int B_M = B->M;

    int C_N = C->N;
    int C_M = C->M;

    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;
    REAL_T *C_value = (REAL_T *) C->value;

    int *csrColIndA = A->csrColInd;
    int *csrColIndB = B->csrColInd;
    int *csrColIndC = C->csrColInd;
    int *csrColIndC_tmp = NULL;
    int *csrRowPtrA = A->csrRowPtr;
    int *csrRowPtrB = B->csrRowPtr;
    int *csrRowPtrC = C->csrRowPtr;
    int *csrRowPtrC_tmp = NULL;
    REAL_T *csrValA = (REAL_T *) A->csrVal;
    REAL_T *csrValB = (REAL_T *) B->csrVal;
    REAL_T *csrValC = (REAL_T *) C->csrVal;
    REAL_T *csrValC_tmp = NULL;

    /* temporary arrays to hold initial C values */
    int *d_ccols = NULL;
    int *d_rptr = NULL;
    REAL_T *d_cvals = NULL;

    REAL_T alpha = (REAL_T) alpha1;
    REAL_T beta = (REAL_T) beta1;
    REAL_T threshold = (REAL_T) threshold1;

    // Variables used for pruning the final matrix
    int nnzC = 0;
    size_t lworkInBytes = 0;
    char *dwork = NULL;


    // rocSPARSE APIs
    rocsparse_status status = rocsparse_status_success;

    rocsparse_operation opA = rocsparse_operation_none;
    rocsparse_operation opB = rocsparse_operation_none;

    rocsparse_datatype computeType = BML_ROCSPARSE_T;

    rocsparse_handle handle = NULL;
    rocsparse_spmat_descr matA, matB, matC, matC_tmp;
    char *dBuffer1 = NULL;      // rocSPARSE needs char * here instead of void *
    size_t bufferSize1 = 0;

    BML_CHECK_ROCSPARSE(rocsparse_create_handle(&handle));

    // convert ellpack to cucsr
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (A);
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (B);
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (C);

    // ensure the matrices are sorted
    TYPED_FUNC(bml_sort_rocsparse_ellpack) (handle,A);
    TYPED_FUNC(bml_sort_rocsparse_ellpack) (handle,B);
    TYPED_FUNC(bml_sort_rocsparse_ellpack) (handle,C);

    // Get total number of nonzero elements for rocsparse calls
    int nnzA, nnzB, nnzC_in;
#pragma omp target map(from:nnzA,nnzB,nnzC_in)
    {
      nnzA = csrRowPtrA[A_N];
      nnzB = csrRowPtrB[B_N];
      nnzC_in = csrRowPtrC[C_N];
    }

    // special case not well handled by cusparse

    if (nnzA == 0 || nnzB == 0)
    {
#pragma omp target teams distribute parallel for \
    shared(nnzC_in, csrValC)
        for (int i = 0; i < nnzC_in; i++)
        {
            csrValC[i] *= beta;
        }
        TYPED_FUNC(bml_cucsr2ellpack_ellpack) (C);
    }
    else
    {

        // Allocate placeholder arrays for storing result of spgemm()
        csrRowPtrC_tmp = (int *) malloc(sizeof(int) * (C_N + 1));
        csrColIndC_tmp = (int *) malloc(sizeof(int) * 1);
        csrValC_tmp = (REAL_T *) malloc(sizeof(REAL_T) * 1);

        // Allocate the same arrays on device
#pragma omp target enter data map(alloc:csrRowPtrC_tmp[:C_N+1], csrColIndC_tmp[:1], csrValC_tmp[:1])

        // Create the rocSPARSE matrices, including C_tmp with placeholder pointers
        // Make these calls within an omp target data region to use device pointers
#pragma omp target data use_device_ptr(csrRowPtrA,csrColIndA,csrValA,	\
				       csrRowPtrB,csrColIndB,csrValB,	\
				       csrRowPtrC,csrColIndC,csrValC,	\
				       csrRowPtrC_tmp,csrColIndC_tmp,csrValC_tmp)
        {
            BML_CHECK_ROCSPARSE(rocsparse_create_csr_descr
                                (&matA,
                                 A_N, A_N, nnzA,
                                 csrRowPtrA, csrColIndA,
                                 csrValA,
                                 rocsparse_indextype_i32,
                                 rocsparse_indextype_i32,
                                 rocsparse_index_base_zero, computeType));
            BML_CHECK_ROCSPARSE(rocsparse_create_csr_descr
                                (&matB,
                                 B_N, B_N, nnzB,
                                 csrRowPtrB, csrColIndB,
                                 csrValB,
                                 rocsparse_indextype_i32,
                                 rocsparse_indextype_i32,
                                 rocsparse_index_base_zero, computeType));
            BML_CHECK_ROCSPARSE(rocsparse_create_csr_descr
                                (&matC, C_N, C_N, nnzC_in, csrRowPtrC,
                                 csrColIndC, csrValC,
                                 rocsparse_indextype_i32,
                                 rocsparse_indextype_i32,
                                 rocsparse_index_base_zero, computeType));
            BML_CHECK_ROCSPARSE(rocsparse_create_csr_descr
                                (&matC_tmp, C_N, C_N, 0, csrRowPtrC_tmp,
                                 csrColIndC_tmp, csrValC_tmp,
                                 rocsparse_indextype_i32,
                                 rocsparse_indextype_i32,
                                 rocsparse_index_base_zero, computeType));
        }

        // Variables used for rocSPARSE debugging calls
        int64_t rows;
        int64_t cols;
        int64_t nnz;
        void *csr_row_ptr;
        void *csr_col_ind;
        void *csr_val;
        rocsparse_indextype row_ptr_type;
        rocsparse_indextype col_ind_type;
        rocsparse_index_base idx_base;
        rocsparse_datatype data_type;
        // spgemm stage 1 = get the working buffer size
        BML_CHECK_ROCSPARSE(rocsparse_spgemm(handle, opA, opB,
                                             &alpha, matA, matB,
                                             &beta, matC, matC_tmp,
                                             computeType,
                                             rocsparse_spgemm_alg_default,
                                             rocsparse_spgemm_stage_buffer_size,
                                             &bufferSize1, NULL));
	// hipDeviceSynchronize(); // Ensure that the previous call is finished
	
        // Allocate the spgemm working buffer
        dBuffer1 = (char *) malloc(sizeof(char) * bufferSize1);
        // Allocate the same array on the device
#pragma omp target enter data map(alloc:dBuffer1[:bufferSize1])

        // spgemm stage 2 = compute # nonzero elements for the spgemm output matrix
#pragma omp target data use_device_ptr(dBuffer1)
        {
            BML_CHECK_ROCSPARSE(rocsparse_spgemm(handle, opA, opB,
                                                 &alpha, matA, matB,
                                                 &beta, matC, matC_tmp,
                                                 computeType,
                                                 rocsparse_spgemm_alg_default,
                                                 rocsparse_spgemm_stage_nnz,
                                                 &bufferSize1, dBuffer1));
        }
	// hipDeviceSynchronize(); // Ensure that the previous call is finished
	
        // Get nnz value returned by spgemm()
        int64_t C_num_rows, C_num_cols, C_nnz_tmp;
        BML_CHECK_ROCSPARSE(rocsparse_spmat_get_size
                            (matC_tmp, &C_num_rows, &C_num_cols, &C_nnz_tmp));
        // Free the placeholder arrays on the device
#pragma omp target exit data map(delete:csrRowPtrC_tmp[:C_N+1], csrColIndC_tmp[:1], csrValC_tmp[:1])
        // Re-allocate the placeholder arrays to create the working arrays on the host
        csrRowPtrC_tmp =
            (int *) realloc(csrRowPtrC_tmp, sizeof(int) * (C_num_rows + 1));
        csrColIndC_tmp =
            (int *) realloc(csrColIndC_tmp, sizeof(int) * C_nnz_tmp);
        csrValC_tmp =
            (REAL_T *) realloc(csrValC_tmp, sizeof(REAL_T) * C_nnz_tmp);

        // Allocate the working arrays on the device
#pragma omp target enter data map(alloc:csrRowPtrC_tmp[:C_num_rows+1], csrColIndC_tmp[:C_nnz_tmp], csrValC_tmp[:C_nnz_tmp])

        // Replace the C_tmp matrix pointers and perform the spgemm computation
#pragma omp target data use_device_ptr(csrRowPtrC_tmp, csrColIndC_tmp, csrValC_tmp,dBuffer1)
        {
            BML_CHECK_ROCSPARSE(rocsparse_csr_set_pointers
                                (matC_tmp, csrRowPtrC_tmp, csrColIndC_tmp,
                                 csrValC_tmp));

/* DEBUG
	BML_CHECK_ROCSPARSE(rocsparse_csr_get(matA, &rows, &cols, &nnz, &csr_row_ptr, &csr_col_ind, &csr_val, &row_ptr_type, &col_ind_type, &idx_base, &data_type));
	for (int i=0;i<10;i++) printf("A[%d]=%f ",i,(((REAL_T *)(csr_val))[i])); printf("\n");
	BML_CHECK_ROCSPARSE(rocsparse_csr_get(matB, &rows, &cols, &nnz, &csr_row_ptr, &csr_col_ind, &csr_val, &row_ptr_type, &col_ind_type, &idx_base, &data_type));
	for (int i=0;i<10;i++) printf("B[%d]=%f ",i,(((REAL_T *)(csr_val))[i])); printf("\n");
	BML_CHECK_ROCSPARSE(rocsparse_csr_get(matC, &rows, &cols, &nnz, &csr_row_ptr, &csr_col_ind, &csr_val, &row_ptr_type, &col_ind_type, &idx_base, &data_type));
	for (int i=0;i<10;i++) printf("C[%d]=%f ",i,(((REAL_T *)(csr_val))[i])); printf("\n");
*/

            // spgemm stage 3 = Perform the computation
            BML_CHECK_ROCSPARSE(rocsparse_spgemm(handle, opA, opB,
                                                 &alpha, matA, matB,
                                                 &beta, matC, matC_tmp,
                                                 computeType,
                                                 rocsparse_spgemm_alg_default,
                                                 rocsparse_spgemm_stage_compute,
                                                 &bufferSize1, dBuffer1));
        }
	// hipDeviceSynchronize(); // Ensure that the previous call is finished

	// Delete the temporary work array
#pragma omp target exit data map(delete:dBuffer1[:bufferSize1])
	// Place the resulting matrix in C
#pragma omp target teams distribute parallel for
	for(int i = 0; i<=C_num_rows; i++) {
	  csrRowPtrC[i] = csrRowPtrC_tmp[i];
	}
#pragma omp target teams distribute parallel for
	for(int i = 0; i<C_nnz_tmp; i++) {
	  csrColIndC[i] = csrColIndC_tmp[i];
	  ((REAL_T *)csrValC)[i] = csrValC_tmp[i];
	}

	// Sort the resulting matrix
	TYPED_FUNC(bml_sort_rocsparse_ellpack) (handle,C);
	BML_CHECK_ROCSPARSE(rocsparse_set_mat_storage_mode((rocsparse_mat_descr)matC, rocsparse_storage_mode_sorted));
	
	// Prune (threshold) the resulting matrix
	TYPED_FUNC(bml_prune_rocsparse_ellpack) (handle,C,threshold);

        // Free the temporary arrays used on the device and host
#pragma omp target exit data map(delete:csrRowPtrC_tmp[:C_num_rows+1],csrColIndC_tmp[:C_nnz_tmp],csrValC_tmp[:C_nnz_tmp])

        free(csrRowPtrC_tmp);
        free(csrColIndC_tmp);
        free(csrValC_tmp);
        free(dBuffer1);

        // Done with matrix multiplication.
        // Update ellpack C matrix (on device): copy from csr to ellpack format
        TYPED_FUNC(bml_cucsr2ellpack_ellpack) (C);

        // Clean up
        BML_CHECK_ROCSPARSE(rocsparse_destroy_spmat_descr(matA));
        BML_CHECK_ROCSPARSE(rocsparse_destroy_spmat_descr(matB));
        BML_CHECK_ROCSPARSE(rocsparse_destroy_spmat_descr(matC));
        BML_CHECK_ROCSPARSE(rocsparse_destroy_spmat_descr(matC_tmp));
    }
    BML_CHECK_ROCSPARSE(rocsparse_destroy_handle(handle));
}

#elif defined(BML_USE_HYPRE)
void TYPED_FUNC(
    bml_multiply_hypre_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha1,
    double beta1,
    double threshold1)
{
    int A_N = A->N;
    int A_M = A->M;

    int B_N = B->N;
    int B_M = B->M;

    int C_N = C->N;
    int C_M = C->M;

    int *csrColIndA = A->csrColInd;
    int *csrColIndB = B->csrColInd;
    int *csrColIndC = C->csrColInd;
    int *csrRowPtrA = A->csrRowPtr;
    int *csrRowPtrB = B->csrRowPtr;
    int *csrRowPtrC = C->csrRowPtr;
    REAL_T *csrValA = (REAL_T *) A->csrVal;
    REAL_T *csrValB = (REAL_T *) B->csrVal;
    REAL_T *csrValC = (REAL_T *) C->csrVal;
    
    /* hypre CSR matrix objects */
    hypre_CSRMatrix  *matA;
    hypre_CSRMatrix  *matB;
    hypre_CSRMatrix  *matC;
        
    REAL_T alpha = (REAL_T) alpha1;
    REAL_T beta = (REAL_T) beta1;

    REAL_T threshold = (REAL_T) threshold1;

    // convert ellpack to cucsr
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (A);
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (B);
    TYPED_FUNC(bml_ellpack2cucsr_ellpack) (C);

    // Create hypre csr matrices A and B
    // Note: The following update is not necessary since the ellpack2cucsr
    // routine updates the csr rowpointers on host and device
//#pragma omp target update from(csrRowPtrA[:A_N+1])
//#pragma omp target update from(csrRowPtrB[:B_N+1])
//#pragma omp target update from(csrRowPtrC[:C_N+1])
    int nnzA = csrRowPtrA[A_N];
    int nnzB = csrRowPtrB[B_N];
    int nnzC_in = csrRowPtrC[C_N];

//     HYPRE_Init();    
//     HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
     int use_vendor = 0;
     int spgemm_alg = 1;
     int spgemm_binned = 0;
     HYPRE_SetSpGemmUseVendor(use_vendor);
     hypre_SetSpGemmAlgorithm(spgemm_alg);
     hypre_SetSpGemmBinned(spgemm_binned);
    /* create hypre csr matrix */
    matA = hypre_CSRMatrixCreate( A_N,A_N,nnzA );
    matB = hypre_CSRMatrixCreate( B_N,B_N,nnzB );
    matC = hypre_CSRMatrixCreate( C_N,C_N,nnzC_in );

#pragma omp target data use_device_ptr(csrRowPtrA,csrColIndA,csrValA, \
		csrRowPtrB,csrColIndB,csrValB, \
		csrRowPtrC,csrColIndC,csrValC)
    {
       hypre_CSRMatrixI(matA) = csrRowPtrA;
       hypre_CSRMatrixJ(matA) = csrColIndA;
       hypre_CSRMatrixData(matA) = csrValA;

       hypre_CSRMatrixI(matB) = csrRowPtrB;
       hypre_CSRMatrixJ(matB) = csrColIndB;
       hypre_CSRMatrixData(matB) = csrValB;

       hypre_CSRMatrixI(matC) = csrRowPtrC;
       hypre_CSRMatrixJ(matC) = csrColIndC;
       hypre_CSRMatrixData(matC) = csrValC;
    }
 
    hypre_CSRMatrix *matD  = hypre_CSRMatrixMultiplyDevice(matA, matB);

    /* add matrices */
    int spadd_use_vendor=0;
    HYPRE_SetSpAddUseVendor(spadd_use_vendor);
    hypre_SetSpAddAlgorithm(1);
    hypre_CSRMatrix *matE = hypre_CSRMatrixAddDevice(alpha, matD, beta, matC);

        // Place the resulting matrix in C
    if (is_above_threshold(threshold, BML_REAL_MIN))
    {
       int nnzE = hypre_CSRMatrixNumNonzeros(matE);
       REAL_T *elmt_tol =
               (REAL_T *) malloc(sizeof(REAL_T) * nnzE);
        // Allocate the working arrays on the device
#pragma omp target enter data map(alloc:elmt_tol[:nnzE])

#pragma omp target teams distribute parallel for
        for(int i = 0; i<nnzE; i++) {
          elmt_tol[i] = threshold;
        }
#pragma omp target data use_device_ptr(elmt_tol)
        {
        hypre_CSRMatrixDropSmallEntriesDevice( matE, threshold, elmt_tol);
        }

#pragma omp target exit data map(delete:elmt_tol[:hypre_CSRMatrixNumNonzeros(matE)])
free(elmt_tol);

    }    

    // Done with matrix multiplication.
    // Update ellpack C matrix (on device): copy from csr to ellpack format
/*
#pragma omp target data use_device_ptr(csrRowPtrC,csrColIndC,csrValC)
{
    hypre_TMemcpy(csrRowPtrC, hypre_CSRMatrixI(matE), HYPRE_Int, C_N + 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
    hypre_TMemcpy(csrColIndC, hypre_CSRMatrixJ(matE), HYPRE_Int, hypre_CSRMatrixNumNonzeros(matE), HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
    hypre_TMemcpy(csrValC, hypre_CSRMatrixData(matE), HYPRE_Real, hypre_CSRMatrixNumNonzeros(matE), HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
}
*/

#pragma omp target data use_device_ptr(csrRowPtrC,csrColIndC,csrValC)
{
    omp_target_memcpy(csrRowPtrC, hypre_CSRMatrixI(matE),
       (C_N + 1) * sizeof(int), 0, 0, 
       omp_get_default_device(), 
       omp_get_default_device());

    omp_target_memcpy(csrColIndC, hypre_CSRMatrixJ(matE),
       hypre_CSRMatrixNumNonzeros(matE) * sizeof(int), 0, 0, 
       omp_get_default_device(), 
       omp_get_default_device());

    omp_target_memcpy(csrValC, hypre_CSRMatrixData(matE),
       hypre_CSRMatrixNumNonzeros(matE) * sizeof(REAL_T), 0, 0, 
       omp_get_default_device(), 
       omp_get_default_device());
}


/*
// DEBUG:
int N1 = hypre_CSRMatrixNumRows(matA);
int C_nnz1 = hypre_CSRMatrixNumNonzeros(matA);
#pragma omp target update from(csrRowPtrC[:N1])
#pragma omp target update from(csrValC[:C_nnz1])
#pragma omp target update from(csrColIndC[:C_nnz1])
for(int k=0; k<N1; k++)
{
   printf("%d, %d, %f \n", csrRowPtrC[k], csrColIndC[k], csrValC[k]);
}
*/
    /* copy from csr to ellpack */
    TYPED_FUNC(bml_cucsr2ellpack_ellpack) (C);

    // destroy hypre data structures
    // Ellpack owns the csr data structure for hypre's matA and matB
    // so we first set hypre's pointers to NULL before destroying the hypre matrices.
    hypre_CSRMatrixI(matA) = NULL;    
    hypre_CSRMatrixJ(matA) = NULL;
    hypre_CSRMatrixData(matA) = NULL;
    hypre_CSRMatrixI(matB) = NULL;
    hypre_CSRMatrixJ(matB) = NULL;
    hypre_CSRMatrixData(matB) = NULL;
    hypre_CSRMatrixI(matC) = NULL;
    hypre_CSRMatrixJ(matC) = NULL;
    hypre_CSRMatrixData(matC) = NULL;
    // destroy 
    hypre_CSRMatrixDestroy(matA);
    hypre_CSRMatrixDestroy(matB);
    hypre_CSRMatrixDestroy(matC);
    hypre_CSRMatrixDestroy(matD);
    hypre_CSRMatrixDestroy(matE);
//    HYPRE_Finalize();
}
#endif
