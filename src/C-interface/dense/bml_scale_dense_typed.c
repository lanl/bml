#ifdef BML_USE_MAGMA
//define boolean data type needed by magma
#include <stdbool.h>
#include "magma_v2.h"
#endif

#include "../../typed.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_scale.h"
#include "../bml_types.h"
#include "bml_allocate_dense.h"
#include "bml_copy_dense.h"
#include "bml_scale_dense.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MKL_GPU
#include "stdio.h"
#include "mkl.h"
#include "mkl_omp_offload.h"
#else
#include "../blas.h"
#endif

/** Scale a dense matrix - result in new matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \return A scaled version of matrix A.
 */
bml_matrix_dense_t *TYPED_FUNC(
    bml_scale_dense_new) (
    void *_scale_factor,
    bml_matrix_dense_t * A)
{
    bml_matrix_dense_t *B = TYPED_FUNC(bml_copy_dense_new) (A);
    REAL_T *scale_factor = _scale_factor;
    REAL_T *B_matrix = B->matrix;
    int myRank = bml_getMyRank();
    int nElems = B->domain->localRowExtent[myRank] * B->ld;
    int startIndex = B->domain->localDispl[myRank];
    int inc = 1;

#ifdef BML_USE_MAGMA
    MAGMA_T scale_factor_ = MAGMACOMPLEX(MAKE) (*scale_factor, 0.);
    MAGMA(scal) (nElems, scale_factor_, B->matrix, inc, bml_queue());
#elif defined (MKL_GPU)
    int dnum = 0;

    REAL_T scale = *scale_factor;
#pragma omp target variant dispatch device(dnum) use_device_ptr(B_matrix)
    G_BLAS(scal) (nElems, MKL_ADDRESS(scale), B_matrix, inc);
#else
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    C_BLAS(SCAL) (&nElems, scale_factor, &(B_matrix[startIndex]), &inc);
#endif
#endif

    return B;
}

/** Scale a dense matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void TYPED_FUNC(
    bml_scale_dense) (
    void *_scale_factor,
    bml_matrix_dense_t * A,
    bml_matrix_dense_t * B)
{
    if (A != B)
    {
        TYPED_FUNC(bml_copy_dense) (A, B);
    }

    REAL_T *scale_factor = _scale_factor;
    REAL_T *B_matrix = B->matrix;
    int myRank = bml_getMyRank();
    int nElems = B->domain->localRowExtent[myRank] * B->ld;
    int startIndex = B->domain->localDispl[myRank];
    int inc = 1;
#ifdef BML_USE_MAGMA
    MAGMA_T scale_factor_ = MAGMACOMPLEX(MAKE) (*scale_factor, 0.);
    MAGMA(scal) (nElems, scale_factor_, B->matrix, inc, bml_queue());
#elif defined (MKL_GPU)
    int dnum = 0;

    REAL_T scale = *scale_factor;
#pragma omp target variant dispatch device(dnum) use_device_ptr(B_matrix)
    G_BLAS(scal) (nElems, MKL_ADDRESS(scale), B_matrix, inc);
#else
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    C_BLAS(SCAL) (&nElems, scale_factor, &(B_matrix[startIndex]), &inc);
#endif
#endif
}

void TYPED_FUNC(
    bml_scale_inplace_dense) (
    void *_scale_factor,
    bml_matrix_dense_t * A)
{
    REAL_T *A_matrix = A->matrix;
    REAL_T *scale_factor = _scale_factor;
    int myRank = bml_getMyRank();
    int nElems = A->domain->localRowExtent[myRank] * A->ld;
    int startIndex = A->domain->localDispl[myRank];
    int inc = 1;
#ifdef BML_USE_MAGMA
    MAGMA_T scale_factor_ = MAGMACOMPLEX(MAKE) (*scale_factor, 0.);
    MAGMA(scal) (nElems, scale_factor_, A->matrix, inc, bml_queue());
#elif defined (MKL_GPU)
    int dnum = 0;

    REAL_T scale = *scale_factor;
#pragma omp target variant dispatch device(dnum) use_device_ptr(A_matrix)
    G_BLAS(scal) (nElems, MKL_ADDRESS(scale), A_matrix, inc);
#else
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    C_BLAS(SCAL) (&nElems, scale_factor, &(A_matrix[startIndex]), &inc);
#endif
#endif
}
