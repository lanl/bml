#include "../../macros.h"
#include "../../typed.h"
#include "../bml_add.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_element_multiply.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_add_ellpack.h"
#include "bml_allocate_ellpack.h"
#include "bml_element_multiply_ellpack.h"
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

/** Element-wise Matrix multiply (Hadamard product)
 *
 * \f$ C_{ij} \leftarrow A_{ij} * B_{ij} \f$
 *
 * \ingroup multiply_group
 *
 * \param A Matrix A
 * \param B Matrix B
 * \param C Matrix C
 * \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_element_multiply_AB_ellpack) (
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

    int all_ix[C_N * num_chunks], all_jx[C_N * num_chunks];
    REAL_T all_x[C_N * num_chunks];

    memset(all_ix, 0, C_N * num_chunks * sizeof(int));
    memset(all_jx, 0, C_N * num_chunks * sizeof(int));
    memset(all_x, 0.0, C_N * num_chunks * sizeof(REAL_T));

#pragma omp target map(to:all_ix[0:C_N*num_chunks],all_jx[0:C_N*num_chunks],all_x[0:C_N*num_chunks])

#endif

#if defined (USE_OMP_OFFLOAD)
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
#pragma omp target teams distribute parallel for \
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
            int k = A_index[ROWMAJOR(i, jp, A_N, A_M)];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                jx[l] = k;
                ix[k] = i + 1;
                l++;
            }

            for (int kp = 0; kp < B_nnz[i]; kp++)
            {
                int kB = B_index[ROWMAJOR(i, kp, B_N, B_M)];
                if (k == kB)
                {
                    x[k] +=
                        A_value[ROWMAJOR(i, jp, A_N, A_M)] *
                        B_value[ROWMAJOR(i, kp, B_N, B_M)];
                }
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
#endif
}
