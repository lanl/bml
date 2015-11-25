#include "../typed.h"
#include "../macros.h"
#include "bml_allocate.h"
#include "bml_gershgorin.h"
#include "bml_types.h"
#include "bml_gershgorin_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

/** Calculate Gershgorin bounds for an ellpack matrix.
 *
 *  \ingroup gershgorin_group
 *
 *  \param A The matrix
 *  \param maxeval Calculated max value
 *  \param maxminusmin Calculated max-min value
 *  \param threshold The matrix threshold 
 */
void TYPED_FUNC(
    bml_gershgorin_ellpack) (
    const bml_matrix_ellpack_t * A,
    double maxeval,
    double maxminusmin,
    const double threshold)
{
    REAL_T radius, absham, dvalue;
    double mineval;
    mineval = 100000000000.0;
    maxeval = -100000000000.0

    int N = A->N;
    int M = A->M;
    int *A_nnz = (int *) A->nnz;
    REAL_T *A_value = (REAL_T *) A->value;

#pragma omp parallel for default(none) shared(N, M, A_nnz, A_value) private(absham, radius, dvalue) reduction(max:maxeval) reduction(min:mineval)
    for (int i = 0; i < N; i++)
    {
        radius = 0.0;

        for (int j = 0; j < A_nnz[i]; j++)
        {
            absham = ABS(A_value[ROWMAJOR(i, j, N, M)]);
            radius += (double) absham;

        }

        radius -= ABS(A_value[ROWMAJOR(i, i, N, M)]);

        dvalue = A_value[ROWMAJOR(i, i, N, M)];

        maxeval = (maxeval > REAL_PART(dvalue + radius) ? maxeval : REAL_PART(dvalue + radius)); 
        mineval = (mineval < REAL_PART(dvalue - radius) ? mineval : REAL_PART(dvalue - radius)); 
    }

    maxminusmin = maxeval - mineval;

}
