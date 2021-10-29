#include "../../macros.h"
#include "../../typed.h"
#include "../bml_add.h"
#include "../bml_allocate.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_add_ellpack.h"
#include "bml_allocate_ellpack.h"
#include "bml_types_ellpack.h"
#include "bml_scale_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Matrix addition.
 *
 * \f$ A = \alpha A + \beta B \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param alpha Scalar factor multiplied by A
 * \param beta Scalar factor multiplied by B
 * \param threshold Threshold for matrix addition
 */
void TYPED_FUNC(
    bml_add_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    double alpha,
    double beta,
    double threshold)
{
    int N = A->N;
    int A_M = A->M;
    int B_M = B->M;

    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    int *B_nnz = B->nnz;
    int *B_index = B->index;

    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];

#if !(defined(__IBMC__) || defined(__ibmxl__) || (defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK))))
    int ix[N], jx[N];
    REAL_T x[N];

    memset(ix, 0, N * sizeof(int));
    memset(jx, 0, N * sizeof(int));
    memset(x, 0.0, N * sizeof(REAL_T));
#endif

#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
    int num_chunks = MIN(OFFLOAD_NUM_CHUNKS, rowMax - rowMin + 1);

    int all_ix[N * num_chunks], all_jx[N * num_chunks];
    REAL_T all_x[N * num_chunks];

    memset(all_ix, 0, N * num_chunks * sizeof(int));
    memset(all_jx, 0, N * num_chunks * sizeof(int));
    memset(all_x, 0.0, N * num_chunks * sizeof(REAL_T));

#pragma omp target map(to:all_ix[0:N*num_chunks],all_jx[0:N*num_chunks],all_x[0:N*num_chunks])

#endif

#if defined (USE_OMP_OFFLOAD)
#if defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__)
#pragma omp teams distribute parallel for \
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int *ix, *jx;
        REAL_T *x;

        ix = &all_ix[chunk * N];
        jx = &all_jx[chunk * N];
        x = &all_x[chunk * N];

#else
#pragma omp target teams distribute parallel for \
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)       \
    firstprivate(ix, jx, x)
#endif
#else
#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for \
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)
#else
#pragma omp parallel for                  \
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)       \
    firstprivate(ix, jx, x)
#endif
#endif
#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
    for (int i = rowMin + chunk; i < rowMax; i = i + num_chunks)
    {
#else
    for (int i = rowMin; i < rowMax; i++)
    {
    
#if defined(__IBMC__) || defined(__ibmxl__)
        int ix[N], jx[N];
        REAL_T x[N];

        memset(ix, 0, N * sizeof(int));
#endif
#endif
        int l = 0;
        if (alpha > (double) 0.0 || alpha < (double) 0.0)
            for (int jp = 0; jp < A_nnz[i]; jp++)
            {
                int k = A_index[ROWMAJOR(i, jp, N, A_M)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    ix[k] = i + 1;
                    jx[l] = k;
                    l++;
                }
                x[k] = x[k] + alpha * A_value[ROWMAJOR(i, jp, N, A_M)];
            }

        if (beta > (double) 0.0 || beta < (double) 0.0)
            for (int jp = 0; jp < B_nnz[i]; jp++)
            {
                int k = B_index[ROWMAJOR(i, jp, N, B_M)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    ix[k] = i + 1;
                    jx[l] = k;
                    l++;
                }
                x[k] = x[k] + beta * B_value[ROWMAJOR(i, jp, N, B_M)];
            }
        A_nnz[i] = l;

        int ll = 0;
        for (int jp = 0; jp < l; jp++)
        {
            int jind = jx[jp];
            REAL_T xTmp = x[jind];
            if (is_above_threshold(xTmp, threshold))
            {
                A_value[ROWMAJOR(i, ll, N, A_M)] = xTmp;
                A_index[ROWMAJOR(i, ll, N, A_M)] = jind;
                ll++;
            }
            x[jind] = 0.0;
            ix[jind] = 0;
        }
        A_nnz[i] = ll;
    }
#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
}
#endif
}

/** Matrix addition.
 *
 * \f$ A = \alpha A + \beta B \f$
 *
 * \ingroup add_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param alpha Scalar factor multiplied by A
 * \param beta Scalar factor multiplied by B
 * \param threshold Threshold for matrix addition
 */
double TYPED_FUNC(
    bml_add_norm_ellpack) (
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    double alpha,
    double beta,
    double threshold)
{
    int N = A->N;
    int A_M = A->M;
    int B_M = B->M;

    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int *A_localRowMin = A->domain->localRowMin;
    int *A_localRowMax = A->domain->localRowMax;

    int *B_nnz = B->nnz;
    int *B_index = B->index;

    int ind, ind2;

    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;

    double trnorm = 0.0;

    int myRank = bml_getMyRank();
    int rowMin = A_localRowMin[myRank];
    int rowMax = A_localRowMax[myRank];

#if !(defined(__IBMC__) || defined(__ibmxl__) || (defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK))))
    int ix[N], jx[N];
    REAL_T x[N];
    REAL_T y[N];

    memset(ix, 0, N * sizeof(int));
    memset(jx, 0, N * sizeof(int));
    memset(x, 0.0, N * sizeof(REAL_T));
    memset(y, 0.0, N * sizeof(REAL_T));
#endif

#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
    int num_chunks = MIN(OFFLOAD_NUM_CHUNKS, rowMax - rowMin + 1);

    int all_ix[N * num_chunks], all_jx[N * num_chunks];
    REAL_T all_x[N * num_chunks], all_y[N * num_chunks];

    memset(all_ix, 0, N * num_chunks * sizeof(int));
    memset(all_jx, 0, N * num_chunks * sizeof(int));
    memset(all_x, 0.0, N * num_chunks * sizeof(REAL_T));
    memset(all_y, 0.0, N * num_chunks * sizeof(REAL_T));

#pragma omp target map(to:all_ix[0:N*num_chunks],all_jx[0:N*num_chunks],all_x[0:N*num_chunks],all_y[0:N*num_chunks]) map(tofrom:trnorm)

#endif

#if defined (USE_OMP_OFFLOAD)
#if defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__)
#pragma omp teams distribute parallel for \
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)       \
    reduction(+:trnorm)
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int *ix, *jx;
        REAL_T *x, *y;

        ix = &all_ix[chunk * N];
        jx = &all_jx[chunk * N];
        x = &all_x[chunk * N];
        y = &all_y[chunk * N];

#else

#pragma omp target teams distribute parallel for	\
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)       \
    firstprivate(ix, jx, x, y)            \
    reduction(+:trnorm)
#endif
#else
#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for                  \
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)       \
    reduction(+:trnorm)
#else
#pragma omp parallel for                  \
    shared(rowMin, rowMax)                \
    shared(A_index, A_value, A_nnz)       \
    shared(B_index, B_value, B_nnz)       \
    firstprivate(ix, jx, x, y)            \
    reduction(+:trnorm)
#endif
#endif
#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
    for (int i = rowMin + chunk; i < rowMax; i = i + num_chunks)
    {
#else
    for (int i = rowMin; i < rowMax; i++)
    {

#if defined(__IBMC__) || defined(__ibmxl__)
        int ix[N], jx[N];
        REAL_T x[N];
        REAL_T y[N];

        memset(ix, 0, N * sizeof(int));
#endif
#endif
        int l = 0;
        for (int jp = 0; jp < A_nnz[i]; jp++)
        {
            int ind = ROWMAJOR(i, jp, N, A_M);
            int k = A_index[ind];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                ix[k] = i + 1;
                y[k] = 0.0;
                //A_index[ROWMAJOR(i, l, N, A_M)] = k;
                jx[l] = k;
                l++;
            }
            x[k] = x[k] + alpha * A_value[ind];
            y[k] = y[k] + A_value[ind];
        }

        for (int jp = 0; jp < B_nnz[i]; jp++)
        {
            int ind = ROWMAJOR(i, jp, N, B_M);
            int k = B_index[ind];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                ix[k] = i + 1;
                y[k] = 0.0;
                jx[l] = k;
                l++;
            }
            x[k] = x[k] + beta * B_value[ind];
            y[k] = y[k] - B_value[ind];
        }
        A_nnz[i] = l;

        int ll = 0;
        for (int jp = 0; jp < l; jp++)
        {
            int jind = jx[jp];
            REAL_T xTmp = x[jind];
            trnorm += y[jind] * y[jind];
            if (is_above_threshold(xTmp, threshold))
            {
                A_value[ROWMAJOR(i, ll, N, A_M)] = xTmp;
                A_index[ROWMAJOR(i, ll, N, A_M)] = jind;
                ll++;
            }
            x[jind] = 0.0;
            ix[jind] = 0;
            y[jind] = 0.0;
        }
        A_nnz[i] = ll;
    }
#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
}
#endif

return trnorm;
}

/** Matrix addition.
 *
 *  A = A + beta * I
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param beta Scalar factor multiplied by I
 *  \param threshold Threshold for matrix addition
 */
void TYPED_FUNC(
    bml_add_identity_ellpack) (
    bml_matrix_ellpack_t * A,
    double beta,
    double threshold)
{
    int N = A->N;
    int A_M = A->M;

    int *A_nnz = A->nnz;
    int *A_index = A->index;
    REAL_T *A_value = (REAL_T *) A->value;

#if !(defined(__IBMC__) || defined(__ibmxl__) || (defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK))))
    int jx[A_M];
    REAL_T x[A_M];

    memset(jx, 0, A_M * sizeof(int));
    memset(x, 0.0, A_M * sizeof(REAL_T));
#endif

#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
    int num_chunks = MIN(OFFLOAD_NUM_CHUNKS, N);

    int all_jx[N * num_chunks];
    REAL_T all_x[N * num_chunks];

    memset(all_jx, 0, N * num_chunks * sizeof(int));
    memset(all_x, 0.0, N * num_chunks * sizeof(REAL_T));

#pragma omp target map(to:all_jx[0:N*num_chunks],all_x[0:N*num_chunks])

#endif

#if defined (USE_OMP_OFFLOAD)
#if defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__)
#pragma omp teams distribute parallel for \
    shared(N, A_M)                \
    shared(A_index, A_value, A_nnz)
    for (int chunk = 0; chunk < num_chunks; chunk++)
    {
        int *jx;
        REAL_T *x;

        jx = &all_jx[chunk * N];
        x = &all_x[chunk * N];

#else
#pragma omp target teams distribute parallel for \
  shared(N, A_M)           \
    shared(A_index, A_value, A_nnz)       \
    firstprivate(jx, x)
#endif
#else
#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for \
  shared(N, A_M)           \
    shared(A_index, A_value, A_nnz)
#else
#pragma omp parallel for                  \
  shared(N, A_M)           \
    shared(A_index, A_value, A_nnz)       \
    firstprivate(jx, x)
#endif
#endif
#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
    for (int i = chunk; i < N; i = i + num_chunks)
    {
#else
    for (int i = 0; i < N; i++)
    {

#if defined(__IBMC__) || defined(__ibmxl__)
        int jx[A_M];
        REAL_T x[A_M];
#endif
#endif
        int l = 0;
        int diag = -1;

        for (int jp = 0; jp < A_nnz[i]; jp++)
        {
            int k = A_index[ROWMAJOR(i, jp, N, A_M)];
            if (k == i)
                diag = jp;
            x[jp] = A_value[ROWMAJOR(i, jp, N, A_M)];
            jx[jp] = k;
            l++;
        }

        if (beta > (double) 0.0 || beta < (double) 0.0)
        {
            // if diagonal entry does not exist
            if (diag == -1)
            {
                x[l] = beta;
                jx[l] = i;
                l++;
            }
            else
            {
                x[diag] = x[diag] + beta;
            }
        }

        int ll = 0;
        for (int jp = 0; jp < l; jp++)
        {
            int jind = jx[jp];
            REAL_T xTmp = x[jp];
            if (is_above_threshold(xTmp, threshold))
            {
                A_value[ROWMAJOR(i, ll, N, A_M)] = xTmp;
                A_index[ROWMAJOR(i, ll, N, A_M)] = jind;
                ll++;
            }
        }
        A_nnz[i] = ll;
    }
#if defined(USE_OMP_OFFLOAD) && (defined(INTEL_SDK) || defined(CRAY_SDK) || defined(__IBMC__) || defined(__ibmxl__))
}
#endif
}

/** Matrix addition.
 *
 *  A = alpha * A + beta * I
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param alpha Scalar factor multiplied by A
 *  \param beta Scalar factor multiplied by I
 *  \param threshold Threshold for matrix addition
 */
void TYPED_FUNC(
    bml_scale_add_identity_ellpack) (
    bml_matrix_ellpack_t * A,
    double alpha,
    double beta,
    double threshold)
{
    // scale then update diagonal
    TYPED_FUNC(bml_scale_inplace_ellpack) (&alpha, A);

    TYPED_FUNC(bml_add_identity_ellpack) (A, beta, threshold);
}
