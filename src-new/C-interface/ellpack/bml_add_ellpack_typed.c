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
    int ix[A->N];
    REAL_T x[A->N];
    REAL_T *A_value = (REAL_T *) A->value;
    REAL_T *B_value = (REAL_T *) B->value;

    memset(ix, 0, A->N * sizeof(int));
    memset(x, 0.0, A->N * sizeof(REAL_T));

    #pragma omp parallel for firstprivate(x,ix)
    for (int i = 0; i < A->N; i++)
    {
        int l = 0;
        for (int jp = 0; jp < A->nnz[i]; jp++)
        {
            int k = A->index[ROWMAJOR(i, jp, A->M)];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                ix[k] = i + 1;
                A->index[ROWMAJOR(i, l, A->M)] = k;
                l++;
            }
            x[k] = x[k] + alpha * A_value[ROWMAJOR(i, jp, A->M)];
        }

        for (int jp = 0; jp < B->nnz[i]; jp++)
        {
            int k = B->index[ROWMAJOR(i, jp, A->M)];
            if (ix[k] == 0)
            {
                x[k] = 0.0;
                ix[k] = i + 1;
                A->index[ROWMAJOR(i, l, A->M)] = k;
                l++;
            }
            x[k] = x[k] + beta * B_value[ROWMAJOR(i, jp, A->M)];
        }
        A->nnz[i] = l;

        int ll = 0;
        for (int jp = 0; jp < l; jp++)
        {
            REAL_T xTmp = x[A->index[ROWMAJOR(i, jp, A->M)]];
            if (is_above_threshold(xTmp, threshold))    // THIS THRESHOLDING COULD BE IGNORED!?
            {
                A_value[ROWMAJOR(i, ll, A->M)] = xTmp;
                A->index[ROWMAJOR(i, ll, A->M)] = A->index[ROWMAJOR(i, jp, A->M)];
                ll++;
            }
            x[A->index[ROWMAJOR(i, jp, A->M)]] = 0.0;
            ix[A->index[ROWMAJOR(i, jp, A->M)]] = 0;
        }
        A->nnz[i] = ll;
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
