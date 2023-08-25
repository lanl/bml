#include "../../typed.h"
#include "../../macros.h"
#include "../blas.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_scale.h"
#include "../bml_types.h"
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
    void *scale_factor,
    bml_matrix_ellpack_t * A)
{
    bml_matrix_ellpack_t *B = TYPED_FUNC(bml_copy_ellpack_new) (A);

    TYPED_FUNC(bml_scale_ellpack)(scale_factor, A, B);

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
    void *_scale_factor,
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B)
{
    // copy necessary so that B has the same structure as A
    if (A != B)
    {
        TYPED_FUNC(bml_copy_ellpack) (A, B);
    }

    REAL_T *scale_factor = _scale_factor;
    REAL_T *B_value = B->value;

    int N = A->N;
    int M = A->M;

#ifdef USE_OMP_OFFLOAD
    REAL_T scale = *scale_factor;
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            B_value[ROWMAJOR(i, j, M, N)] =
                scale * B_value[ROWMAJOR(i, j, M, N)];
        }
    }
#else // offload conditional

#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    int myRank = bml_getMyRank();
    int nElems = B->domain->localRowExtent[myRank] * B->M;
    int startIndex = B->domain->localDispl[myRank];
    int inc = 1;
    C_BLAS(SCAL) (&nElems, scale_factor, &(B_value[startIndex]), &inc);
#endif

#endif // offload conditional
}

void TYPED_FUNC(
    bml_scale_inplace_ellpack) (
    void *_scale_factor,
    bml_matrix_ellpack_t * A)
{
    REAL_T *scale_factor = _scale_factor;
    REAL_T *A_value = A->value;

#ifdef USE_OMP_OFFLOAD
    int N = A->N;
    int M = A->M;

    REAL_T scale = *scale_factor;
    size_t MbyN = N * M;
#pragma omp target teams distribute parallel for map(to:MbyN,scale)
    for (size_t i = 0; i < MbyN; i++)
    {
      A_value[i] = scale * A_value[i];
    }
#else // offload conditional

#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    int myRank = bml_getMyRank();
    int number_elements = A->domain->localRowExtent[myRank] * A->M;
    int startIndex = A->domain->localDispl[myRank];
    int inc = 1;
    C_BLAS(SCAL) (&number_elements, scale_factor, &(A_value[startIndex]),
                  &inc);
#endif

#endif // offload conditional

}
