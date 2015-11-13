#include "../macros.h"
#include "../typed.h"
#include "bml_allocate.h"
#include "bml_add.h"
#include "bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_add_ellpack.h"
#include "bml_types_ellpack.h"

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
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B,
    const double alpha,
    const double beta,
    const double threshold)
{
    int N = A->N;
    int M = A->M;
    int ix[N];
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int *B_nnz = B->nnz;
    int *B_index = B->index;
    REAL_T x[N];
    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;

    memset(ix, 0, N * sizeof(int));
    memset(x, 0.0, N * sizeof(REAL_T));

#pragma omp parallel for default(none) firstprivate(x, ix) shared(N, M, A_index, A_value, A_nnz, B_index, B_value, B_nnz)
    for (int i = 0; i < N; i++)
    {
        int l = 0;
        for (int jp = 0; jp < A_nnz[i]; jp++)
        {
            int k = A_index[ROWMAJOR(i, jp, M)];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                ix[k] = i + 1;
                A_index[ROWMAJOR(i, l, M)] = k;
                l++;
            }
            x[k] = x[k] + alpha * A_value[ROWMAJOR(i, jp, M)];
        }

        for (int jp = 0; jp < B_nnz[i]; jp++)
        {
            int k = B_index[ROWMAJOR(i, jp, M)];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                ix[k] = i + 1;
                A_index[ROWMAJOR(i, l, M)] = k;
                l++;
            }
            x[k] = x[k] + beta * B_value[ROWMAJOR(i, jp, M)];
        }
        A_nnz[i] = l;

        int ll = 0;
        for (int jp = 0; jp < l; jp++)
        {
            REAL_T xTmp = x[A_index[ROWMAJOR(i, jp, M)]];
            if (is_above_threshold(xTmp, threshold))    // THIS THRESHOLDING COULD BE IGNORED!?
            {
                A_value[ROWMAJOR(i, ll, M)] = xTmp;
                A_index[ROWMAJOR(i, ll, M)] = A_index[ROWMAJOR(i, jp, M)];
                ll++;
            }
            x[A_index[ROWMAJOR(i, jp, M)]] = 0.0;
            ix[A_index[ROWMAJOR(i, jp, M)]] = 0;
        }
        A_nnz[i] = ll;
    }
}

/** Matrix addition.
 *
 *  A = A + beta * I
 *
 *  \ingroup add_group
 *
 *  \param A Matrix A
 *  \param beta Scalar factor multiplied by A
 *  \param threshold Threshold for matrix addition
 */
void TYPED_FUNC(
    bml_add_identity_ellpack) (
    const bml_matrix_ellpack_t * A,
    const double beta,
    const double threshold)
{
    REAL_T alpha = (REAL_T) 1.0;

    bml_matrix_ellpack_t *Id =
        TYPED_FUNC(bml_identity_matrix_ellpack) (A->N, A->M);

    TYPED_FUNC(bml_add_ellpack) (A, Id, alpha, beta, threshold);

    bml_deallocate_ellpack(Id);
}
