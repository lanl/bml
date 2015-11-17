#include "../macros.h"
#include "../typed.h"
#include "bml_add.h"
#include "bml_allocate.h"
#include "bml_multiply.h"
#include "bml_types.h"
#include "bml_add_ellpack.h"
#include "bml_allocate_ellpack.h"
#include "bml_multiply_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
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
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double alpha,
    const double beta,
    const double threshold)
{
    REAL_T salpha = (REAL_T) alpha;
    REAL_T sbeta = (REAL_T) beta;
    REAL_T sthreshold = (REAL_T) threshold;

    bml_matrix_ellpack_t *A2 =
        TYPED_FUNC(bml_zero_matrix_ellpack) (A->N, A->M);

    if (A != NULL && A == B)
    {
        TYPED_FUNC(bml_multiply_x2_ellpack) (A, A2, sthreshold);
    }
    else
    {
        TYPED_FUNC(bml_multiply_AB_ellpack) (B, A, A2, sthreshold);
    }

    TYPED_FUNC(bml_add_ellpack) (C, A2, sbeta, salpha, sthreshold);

    bml_deallocate_ellpack(A2);
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
void TYPED_FUNC(
    bml_multiply_x2_ellpack) (
    const bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    const double threshold)
{
    int N = X->N;
    int M = X->M;
    int ix[N];

    int *X_index = X->index;
    int *X2_index = X2->index;
    int *X_nnz = X->nnz;
    int *X2_nnz = X2->nnz;

    REAL_T x[N];
    REAL_T traceX = 0.0;
    REAL_T traceX2 = 0.0;
    REAL_T *X_value = (REAL_T *) X->value;
    REAL_T *X2_value = (REAL_T *) X2->value;

    memset(ix, 0, N * sizeof(int));
    memset(x, 0.0, N * sizeof(REAL_T));

#pragma omp parallel for \
    default(none) \
    firstprivate(ix, x) \
    shared(N, M, X_index, X_value, X_nnz, X2_index, X2_value, X2_nnz) \
    reduction(+: traceX, traceX2)
    for (int i = 0; i < N; i++) // CALCULATES THRESHOLDED X^2
    {
        int l = 0;
        for (int jp = 0; jp < X_nnz[i]; jp++)
        {
            REAL_T a = X_value[ROWMAJOR(i, jp, N, M)];
            int j = X_index[ROWMAJOR(i, jp, N, M)];
            if (j == i)
            {
                traceX = traceX + a;
            }
            for (int kp = 0; kp < X_nnz[j]; kp++)
            {
                int k = X_index[ROWMAJOR(j, kp, N, M)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    X2_index[ROWMAJOR(i, l, N, M)] = k;
                    ix[k] = i + 1;
                    l++;
                }
                x[k] = x[k] + a * X_value[ROWMAJOR(j, kp, N, M)];       // TEMPORARY STORAGE VECTOR LENGTH FULL N
            }
        }

        // Check for number of non-zeroes per row exceeded
        if (l > M)
        {
            printf("\nERROR: Number of non-zeroes per row > M, Increase M\n");
            exit(-1);
        }

        int ll = 0;
        for (int j = 0; j < l; j++)
        {
            int jp = X2_index[ROWMAJOR(i, j, N, M)];
            REAL_T xtmp = x[jp];
            // The diagonal elements are stored in the first column
            if (jp == i)
            {
                traceX2 = traceX2 + xtmp;
                X2_value[ROWMAJOR(i, ll, N, M)] = xtmp;
                X2_index[ROWMAJOR(i, ll, N, M)] = jp;
                ll++;
            }
            else if (is_above_threshold(xtmp, threshold))
            {
                X2_value[ROWMAJOR(i, ll, N, M)] = xtmp;
                X2_index[ROWMAJOR(i, ll, N, M)] = jp;
                ll++;
            }
            ix[jp] = 0;
            x[jp] = 0.0;
        }
        X2_nnz[i] = ll;
    }
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
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    const double threshold)
{
    int N = A->N;
    int M = A->M;
    int ix[N];

    REAL_T x[N];
    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;
    REAL_T *C_value = (REAL_T *) C->value;

    int *A_nnz = A->nnz;
    int *B_nnz = B->nnz;
    int *C_nnz = C->nnz;

    int *A_index = A->index;
    int *B_index = B->index;
    int *C_index = C->index;

    memset(ix, 0, N * sizeof(int));
    memset(x, 0.0, N * sizeof(REAL_T));

#pragma omp parallel for \
    default(none) \
    firstprivate(ix, x) \
    shared(N, M, A_index, A_value, A_nnz, B_index, B_value, B_nnz, C_index, C_value, C_nnz)
    for (int i = 0; i < N; i++)
    {
        int l = 0;
        for (int jp = 0; jp < A_nnz[i]; jp++)
        {
            REAL_T a = A_value[ROWMAJOR(i, jp, N, M)];
            int j = A_index[ROWMAJOR(i, jp, N, M)];

            for (int kp = 0; kp < B_nnz[j]; kp++)
            {
                int k = B_index[ROWMAJOR(j, kp, N, M)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    C_index[ROWMAJOR(i, l, N, M)] = k;
                    ix[k] = i + 1;
                    l++;
                }
                x[k] = x[k] + a * B_value[ROWMAJOR(j, kp, N, M)];       // TEMPORARY STORAGE VECTOR LENGTH FULL N
            }
        }

        // Check for number of non-zeroes per row exceeded
        if (l > M)
        {
            printf("\nERROR: Number of non-zeroes per row > M, Increase M\n");
            exit(-1);
        }

        int ll = 0;
        for (int j = 0; j < l; j++)
        {
            int jp = C_index[ROWMAJOR(i, j, N, M)];
            REAL_T xtmp = x[jp];
            // Diagonal elements are saved in first column
            if (jp == i)
            {
                C_value[ROWMAJOR(i, ll, N, M)] = xtmp;
                C_index[ROWMAJOR(i, ll, N, M)] = jp;
                ll++;
            }
            else if (is_above_threshold(xtmp, threshold))
            {
                C_value[ROWMAJOR(i, ll, N, M)] = xtmp;
                C_index[ROWMAJOR(i, ll, N, M)] = jp;
                ll++;
            }
            ix[jp] = 0;
            x[jp] = 0.0;
        }
        C_nnz[i] = ll;
    }
}
