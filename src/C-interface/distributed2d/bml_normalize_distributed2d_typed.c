#include "../../macros.h"
#include "../../typed.h"
#include "../bml_types.h"
#include "../bml_allocate.h"
#include "../bml_normalize.h"
#include "../bml_getters.h"
#include "../bml_scale.h"
#include "../bml_introspection.h"
#include "../bml_add.h"
#include "bml_normalize_distributed2d.h"
#include "bml_types_distributed2d.h"

#include <float.h>
#include <complex.h>

void TYPED_FUNC(
    bml_normalize_distributed2d) (
    bml_matrix_distributed2d_t * A,
    double mineval,
    double maxeval)
{
    double maxminusmin = maxeval - mineval;
    double gershfact = maxeval / maxminusmin;
    REAL_T scalar = (REAL_T) - 1.0 / maxminusmin;
    double threshold = 0.0;

    bml_scale_inplace(&scalar, bml_get_local_matrix(A));
    if (A->myprow == A->mypcol)
        bml_add_identity(bml_get_local_matrix(A), gershfact, threshold);
}

void *TYPED_FUNC(
    bml_gershgorin_distributed2d) (
    bml_matrix_distributed2d_t * A)
{
    int nloc = A->N / A->nprows;
    REAL_T *rad = bml_allocate_memory(nloc * sizeof(REAL_T));
    REAL_T *offdiag_sum =
        bml_accumulate_offdiag(A->matrix, A->myprow != A->mypcol);
    MPI_Allreduce(offdiag_sum, rad, nloc, MPI_T, MPI_SUM, A->row_comm);
    free(offdiag_sum);

    double emin = DBL_MAX;
    double emax = DBL_MIN;

    if (A->myprow == A->mypcol)
    {
        REAL_T *dval = bml_get_diagonal(A->matrix);

        for (int i = 0; i < nloc; i++)
        {
            if (REAL_PART(dval[i] + rad[i]) > emax)
                emax = REAL_PART(dval[i] + rad[i]);
            if (REAL_PART(dval[i] - rad[i]) < emin)
                emin = REAL_PART(dval[i] - rad[i]);
        }
        free(dval);
    }
    bml_free_memory(rad);

    double *eval = bml_allocate_memory(sizeof(double) * 2);

    MPI_Allreduce(&emin, eval, 1, MPI_DOUBLE, MPI_MIN, A->comm);
    MPI_Allreduce(&emax, eval + 1, 1, MPI_DOUBLE, MPI_MAX, A->comm);

    return eval;
}
