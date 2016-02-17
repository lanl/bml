#include "../macros.h"
#include "../typed.h"
#include "bml_allocate.h"
#include "bml_normalize.h"
#include "bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_normalize_ellpack.h"
#include "bml_scale_ellpack.h"
#include "bml_add_ellpack.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Normalize ellpack matrix given Gershgorin bounds.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  \param maxeval Calculated max value
 *  \param maxminusmin Calculated max-min value
 */
void TYPED_FUNC(
    bml_normalize_ellpack) (
    bml_matrix_ellpack_t * A,
    const double maxeval,
    const double maxminusmin)
{
    double gershfact = maxeval / maxminusmin;
    double scalar = (double) -1.0 / maxminusmin;
    double threshold = 0.0;

    bml_scale_inplace_ellpack(scalar, A);
    bml_add_identity_ellpack(A, gershfact, threshold);
}

/** Calculate Gershgorin bounds for an ellpack matrix.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  returns maxeval Calculated max value
 *  returns maxminusmin Calculated max-min value
 */
void *TYPED_FUNC(
    bml_gershgorin_ellpack) (
    const bml_matrix_ellpack_t * A)
{
    REAL_T radius, absham, dvalue;
    double emin = 100000000000.0;
    double emax = -100000000000.0;

    double *eval = bml_allocate_memory(sizeof(double) * 2);

    int N = A->N;
    int M = A->M;
    int *A_nnz = (int *) A->nnz;
    int *A_index = (int *) A->index;
    REAL_T *A_value = (REAL_T *) A->value;

#pragma omp parallel for default(none) \
    shared(N, M, A_nnz, A_index, A_value) \
    private(absham, radius, dvalue) \
    reduction(max:emax) \
    reduction(min:emin)
    for (int i = 0; i < N; i++)
    {
        radius = 0.0;

        for (int j = 0; j < A_nnz[i]; j++)
        {
            if (i == A_index[ROWMAJOR(i, j, N, M)])
                dvalue = A_value[ROWMAJOR(i, j, N, M)];
            else
            {
                absham = ABS(A_value[ROWMAJOR(i, j, N, M)]);
                radius += (double) absham;
            }

        }

        emax =
            (emax >
             REAL_PART(dvalue + radius) ? emax : REAL_PART(dvalue + radius));
        emin =
            (emin <
             REAL_PART(dvalue - radius) ? emin : REAL_PART(dvalue - radius));
    }

    eval[0] = emax;
    eval[1] = emax - emin;

    return eval;
}
