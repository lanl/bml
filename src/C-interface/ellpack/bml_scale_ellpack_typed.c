#include "../../typed.h"
#include "../blas.h"
#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_copy_ellpack.h"
#include "bml_logger.h"
#include "bml_parallel.h"
#include "bml_scale.h"
#include "bml_scale_ellpack.h"
#include "bml_types.h"
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
    const REAL_T * scale_factor,
    const bml_matrix_ellpack_t * A)
{
    bml_matrix_ellpack_t *B = TYPED_FUNC(bml_copy_ellpack_new) (A);

    REAL_T *B_value = B->value;
    int myRank = bml_getMyRank();
    int nElems = B->domain->localRowExtent[myRank] * B->M;
    int startIndex = B->domain->localDispl[myRank];
    int inc = 1;

    int N = A->N;
    int M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int *A_value = A->value;

#pragma omp target update from(A_nnz[:N], A_index[:N*M], A_value[:N*M])

#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    C_BLAS(SCAL) (&nElems, scale_factor, &(B_value[startIndex]), &inc);
#endif

    // push result to GPU
    int *B_nnz = B->nnz;
    int *B_index = B->index;

#pragma omp target update to(B_nnz[:N], B_index[:N*M], B_value[:N*M])

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
    const REAL_T * scale_factor,
    const bml_matrix_ellpack_t * A,
    const bml_matrix_ellpack_t * B)
{
    if (A != B)
        TYPED_FUNC(bml_copy_ellpack) (A, B);

    REAL_T *B_value = B->value;
    int myRank = bml_getMyRank();
    int nElems = B->domain->localRowExtent[myRank] * B->M;
    int startIndex = B->domain->localDispl[myRank];
    int inc = 1;

    int N = A->N;
    int M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;
    int *A_value = A->value;

#pragma omp target update from(A_nnz[:N], A_index[:N*M], A_value[:N*M])

#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    C_BLAS(SCAL) (&nElems, scale_factor, &(B_value[startIndex]), &inc);
#endif

    // push result to GPU
    int *B_nnz = B->nnz;
    int *B_index = B->index;

#pragma omp target update to(B_nnz[:N], B_index[:N*M], B_value[:N*M])

}

void TYPED_FUNC(
    bml_scale_inplace_ellpack) (
    const REAL_T * scale_factor,
    bml_matrix_ellpack_t * A)
{
    REAL_T *A_value = A->value;

    int myRank = bml_getMyRank();
    int number_elements = A->domain->localRowExtent[myRank] * A->M;
    int startIndex = A->domain->localDispl[myRank];
    int inc = 1;

    int N = A->N;
    int M = A->M;
    int *A_nnz = A->nnz;
    int *A_index = A->index;

#pragma omp target update from(A_nnz[:N], A_index[:N*M], A_value[:N*M])

#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    C_BLAS(SCAL) (&number_elements, scale_factor, &(A_value[startIndex]),
                  &inc);
#endif

    // push result to GPU

#pragma omp target update to(A_nnz[:N], A_index[:N*M], A_value[:N*M])

}
