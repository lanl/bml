#include "../macros.h"
#include "../typed.h"
#include "bml_allocate.h"
#include "bml_normalize.h"
#include "bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_normalize_dense.h"
#include "bml_scale_dense.h"
#include "bml_add_dense.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/** Normalize dense matrix given Gershgorin bounds.
 *
 *  \ingroup normalize_group
 *
 *  \param A The matrix
 *  \param maxeval Calculated max value
 *  \param maxminusmin Calculated max-min value
 */
void TYPED_FUNC(
    bml_normalize_dense) (
    bml_matrix_dense_t * A,
    const double maxeval,
    const double maxminusmin)
{
    double gershfact = maxeval / maxminusmin;
    double scalar = (double)-1.0 / maxminusmin;

    bml_scale_inplace_dense(scalar, A);
    bml_add_identity_dense(A, gershfact);
}

/** Calculate Gershgorin bounds for a dense matrix.
 *
 *  \ingroup gershgorin_group
 *
 *  \param A The matrix
 *  returns maxeval Calculated max value
 *  returns maxminusmin Calculated max-min value
 */
void *TYPED_FUNC(
    bml_gershgorin_dense) (
    const bml_matrix_dense_t * A)
{
    REAL_T radius, dvalue, absham;

    int N = A->N;
    REAL_T *A_matrix = A->matrix;

    double emin = 100000000000.0;
    double emax = -100000000000.0;

    double *eval = bml_allocate_memory(sizeof(double) * 2);

#pragma omp parallel for \
    default(none) \
    shared(N, A_matrix) \
    private(absham, radius, dvalue) \
    reduction(max:emax) \
    reduction(min:emin)
    for (int i = 0; i < N; i++)
    {
        radius = 0.0;

        for (int j = 0; j < N; j++)
        {
            absham = ABS(A_matrix[ROWMAJOR(i, j, N, N)]);
            radius += (double) absham;
        }

        dvalue = A_matrix[ROWMAJOR(i, i, N, N)];

        radius -= ABS(dvalue);

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
