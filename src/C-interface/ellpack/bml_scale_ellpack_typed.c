#include "../typed.h"
#include "../blas.h"
#include "bml_allocate.h"
#include "bml_scale.h"
#include "bml_parallel.h"
#include "bml_types.h"
#include "bml_allocate_ellpack.h"
#include "bml_copy_ellpack.h"
#include "bml_scale_ellpack.h"
#include "bml_types_ellpack.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Scale an ellpack matrix - result is a new matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \return A scale version of matrix A.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_scale_ellpack_new) (
    const double scale_factor,
    const bml_matrix_ellpack_t * A)
{
    REAL_T sfactor = (REAL_T) scale_factor;

    bml_matrix_ellpack_t *B = TYPED_FUNC(bml_copy_ellpack_new) (A);

    REAL_T * B_value = B->value;
    int myRank = bml_getMyRank();
    //int nElems = B->N * B->M;
    int nElems = B->domain->localRowExtent[myRank] * B->M;
    int startIndex = B->domain->localDispl[myRank];
    int inc = 1;

#if DO_MPI
    if (bml_getNRanks() > 1 && B->distribution_mode == distributed)
    { 
      C_BLAS(SCAL) (&nElems, &sfactor, &(B_value[startIndex]), &inc);
    }
#else
    C_BLAS(SCAL) (&nElems, &sfactor, B->value, &inc);
#endif

    return B;
}

/** Scale an ellpack matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void TYPED_FUNC(
    bml_scale_ellpack) (
    const double scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B)
{
    REAL_T sfactor = (REAL_T) scale_factor;

    if (A != B)
        TYPED_FUNC(bml_copy_ellpack) (A, B);

    REAL_T * B_value = B->value;
    int myRank = bml_getMyRank();
    //int nElems = B->N * B->M;
    int nElems = B->domain->localRowExtent[myRank] * B->M;
    int startIndex = B->domain->localDispl[myRank];
    int inc = 1;

    C_BLAS(SCAL) (&nElems, &sfactor, &(B_value[startIndex]), &inc);
    //C_BLAS(SCAL) (&nElems, &sfactor, B->value, &inc);
}

void TYPED_FUNC(
    bml_scale_inplace_ellpack) (
    const double scale_factor,
    bml_matrix_ellpack_t * A)
{
    REAL_T * A_value = A->value;
    REAL_T scale_factor_ = (REAL_T) scale_factor;

    int myRank = bml_getMyRank();
    //int number_elements = A->N * A->M;
    int number_elements = A->domain->localRowExtent[myRank] * A->M;
    int startIndex = A->domain->localDispl[myRank];
    int inc = 1;

    C_BLAS(SCAL) (&number_elements, &scale_factor_, &(A_value[startIndex]), &inc);
    //C_BLAS(SCAL) (&number_elements, &scale_factor_, A->value, &inc);
}
