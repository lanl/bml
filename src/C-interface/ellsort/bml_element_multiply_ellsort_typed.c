#include "../../macros.h"
#include "../../typed.h"
#include "../bml_add.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_element_multiply.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_add_ellsort.h"
#include "bml_allocate_ellsort.h"
#include "bml_element_multiply_ellsort.h"
#include "bml_types_ellsort.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    bml_element_multiply_AB_ellsort) (
    bml_matrix_ellsort_t * A,
    bml_matrix_ellsort_t * B,
    bml_matrix_ellsort_t * C,
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

#if !(defined(__IBMC__) || defined(__ibmxl__))
    int ix[C->N], jx[C->N];
    REAL_T x[C->N];

    memset(ix, 0, C->N * sizeof(int));
    memset(jx, 0, C->N * sizeof(int));
    memset(x, 0.0, C->N * sizeof(REAL_T));
#endif

#if defined(__IBMC__) || defined(__ibmxl__)
#pragma omp parallel for                     \
  shared(A_N, A_M, A_nnz, A_index, A_value)  \
  shared(A_localRowMin, A_localRowMax)       \
  shared(B_N, B_M, B_nnz, B_index, B_value)  \
  shared(C_N, C_M, C_nnz, C_index, C_value)  \
  shared(myRank)
#else
#pragma omp parallel for                     \
  shared(A_N, A_M, A_nnz, A_index, A_value)  \
  shared(A_localRowMin, A_localRowMax)       \
  shared(B_N, B_M, B_nnz, B_index, B_value)  \
  shared(C_N, C_M, C_nnz, C_index, C_value)  \
  shared(myRank)                             \
  firstprivate(ix, jx, x)
#endif

    //for (int i = 0; i < A_N; i++)
    for (int i = A_localRowMin[myRank]; i < A_localRowMax[myRank]; i++)
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
            LOG_ERROR("Number of non-zeroes per row > M, Increase M\n");
        }

        int ll = 0;
        for (int j = 0; j < l; j++)
        {
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
}
