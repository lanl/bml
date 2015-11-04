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
 * C = alpha * A * B + beta * C
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 *  \param alpha Scalar factor multiplied by A * B
 *  \param beta Scalar factor multiplied by C
 *  \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_multiply_ellpack) (
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    const bml_matrix_ellpack_t * C,
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
 * X2 = X * X
 *
 *  \ingroup multiply_group
 *
 *  \param X Matrix X
 *  \param X2 Matrix X2
 *  \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_multiply_x2_ellpack) (
    const bml_matrix_ellpack_t * X,
    const bml_matrix_ellpack_t * X2,
    const double threshold)
{
    int hsize = X->N;
    int msize = X->M;
    int ix[hsize];

    int *X_index = X->index;
    int *X2_index = X2->index;
    int *X_nnz = X->nnz;
    int *X2_nnz = X2->nnz;

    REAL_T x[hsize];
    REAL_T traceX = 0.0;
    REAL_T traceX2 = 0.0;
    REAL_T *X_value = (REAL_T *) X->value;
    REAL_T *X2_value = (REAL_T *) X2->value;

    memset(ix, 0, hsize * sizeof(int));
    memset(x, 0.0, hsize * sizeof(REAL_T));

    #pragma omp parallel for firstprivate(ix,x) reduction(+:traceX,traceX2)
    for (int i = 0; i < hsize; i++)     // CALCULATES THRESHOLDED X^2
    {
        int l = 0;
        for (int jp = 0; jp < X_nnz[i]; jp++)
        {
            REAL_T a = X_value[ROWMAJOR(i, jp, msize)];
            int j = X_index[ROWMAJOR(i, jp, msize)];
            if (j == i)
            {
                traceX = traceX + a;
            }
            for (int kp = 0; kp < X_nnz[j]; kp++)
            {
                int k = X_index[ROWMAJOR(j, kp, msize)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    X2_index[ROWMAJOR(i, l, msize)] = k;
                    ix[k] = i + 1;
                    l++;
                }
                x[k] = x[k] + a * X_value[ROWMAJOR(j, kp, msize)];      // TEMPORARY STORAGE VECTOR LENGTH FULL N
            }
        }

        // Check for number of non-zeroes per row exceeded
        if (l > msize)
        {
            printf("\nERROR: Number of non-zeroes per row > M, Increase M\n");
            exit(-1);
        }

        int ll = 1;
        for (int j = 0; j < l; j++)
        {
            int jp = X2_index[ROWMAJOR(i, j, msize)];
            REAL_T xtmp = x[jp];
            // The diagonal elements are stored in the first column
            if (jp == i)
            {
                traceX2 = traceX2 + xtmp;
                X2_value[ROWMAJOR(i, 0, msize)] = xtmp;
                X2_index[ROWMAJOR(i, 0, msize)] = jp;
            }
            else if (is_above_threshold(xtmp, threshold))
            {
                X2_value[ROWMAJOR(i, ll, msize)] = xtmp;
                X2_index[ROWMAJOR(i, ll, msize)] = jp;
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
 * C = B * A
 *
 *  \ingroup multiply_group
 *
 *  \param A Matrix A
 *  \param B Matrix B
 *  \param C Matrix C
 *  \param threshold Used for sparse multiply
 */
void TYPED_FUNC(
    bml_multiply_AB_ellpack) (
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    const bml_matrix_ellpack_t * C,
    const double threshold)
{
    int hsize = A->N;
    int msize = A->M;
    int ix[hsize];

    REAL_T x[hsize];
    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;
    REAL_T *C_value = (REAL_T *) C->value;

    int *A_nnz = A->nnz;
    int *B_nnz = B->nnz;
    int *C_nnz = C->nnz;

    int *A_index = A->index;
    int *B_index = B->index;
    int *C_index = C->index;

    memset(ix, 0, hsize * sizeof(int));
    memset(x, 0.0, hsize * sizeof(REAL_T));

    #pragma omp parallel for firstprivate(ix,x)
    for (int i = 0; i < hsize; i++)
    {
        int l = 0;
        for (int jp = 0; jp < A_nnz[i]; jp++)
        {
            REAL_T a = A_value[ROWMAJOR(i, jp, msize)];
            int j = A_index[ROWMAJOR(i, jp, msize)];

            for (int kp = 0; kp < B_nnz[j]; kp++)
            {
                int k = B_index[ROWMAJOR(j, kp, msize)];
                if (ix[k] == 0)
                {
                    x[k] = 0.0;
                    C_index[ROWMAJOR(i, l, msize)] = k;
                    ix[k] = i + 1;
                    l++;
                }
                x[k] = x[k] + a * B_value[ROWMAJOR(j, kp, msize)];      // TEMPORARY STORAGE VECTOR LENGTH FULL N
            }
        }

        // Check for number of non-zeroes per row exceeded
        if (l > msize)
        {
            printf("\nERROR: Number of non-zeroes per row > M, Increase M\n");
            exit(-1);
        }

        int ll = 1;
        for (int j = 0; j < l; j++)
        {
            int jp = C_index[ROWMAJOR(i, j, msize)];
            REAL_T xtmp = x[jp];
            // Diagonal elements are saved in first column
            if (jp == i)
            {
                C_value[ROWMAJOR(i, 0, msize)] = xtmp;
                C_index[ROWMAJOR(i, 0, msize)] = jp;
            }
            else if (is_above_threshold(xtmp, threshold))
            {
                C_value[ROWMAJOR(i, ll, msize)] = xtmp;
                C_index[ROWMAJOR(i, ll, msize)] = jp;
                ll++;
            }
            ix[jp] = 0;
            x[jp] = 0.0;
        }
        C_nnz[i] = ll;
    }
}
