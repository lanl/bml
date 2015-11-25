#include "../typed.h"
#include "../macros.h"
#include "bml_gershgorin.h"
#include "bml_types.h"
#include "bml_gershgorin_dense.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

/** Calculate Gershgorin bounds for a dense matrix.
 *
 *  \ingroup gershgorin_group
 *
 *  \param A The matrix
 *  \param maxeval Calculated max value
 *  \param maxminusmin Calculated max-min value
 *  \param threshold The matrix threshold
 */
void TYPED_FUNC(
    bml_gershgorin_dense) (
    const bml_matrix_dense_t * A,
    double maxeval,
    double maxminusmin,
    const double threshold)
{
    REAL_T radius, dvalue;
    int nnz = 0;

    int N = A->N;
    REAL_T *A_matrix = A->matrix;

    double mineval = 100000000000.0;
    maxeval = -100000000000.0

#pragma omp parallel for default(none) shared(N, A_matrix) private(absham, radius, dvalue) reduction(max:maxeval) reduction(min:mineval)
    for (int i = 0; i < N; i++)
    {
        radius = 0.0;

        for (int j = 0; j < N; j++)
        {
            absham = ABS(A_matrix[ROWMAJOR(i, j, N, N)]);
            radius += (double)absham;

            if (absham > threshold) nnz += 1;
        }

        radius -= ABS(A_matrix[ROWMAJOR(i, i, N, N)]);

        dvalue = A_matrix[ROWMAJOR(i, i, N, N)];

        maxeval = (maxeval > (dvalue + radius) ? maxeval : dvalue + radius);
        mineval = (mineval < (dvalue - radius) ? maxeval : dvalue - radius);
    }
 
    maxminusmin = maxeval - mineval;
}
