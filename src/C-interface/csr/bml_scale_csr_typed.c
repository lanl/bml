#include "../../typed.h"
#include "../blas.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_scale.h"
#include "../bml_types.h"
#include "bml_allocate_csr.h"
#include "bml_copy_csr.h"
#include "bml_scale_csr.h"
#include "bml_types_csr.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Scale an csr matrix row.
 *
 *  \ingroup scale_group
 *
 *  \param arow The row to be scaled
 */
void TYPED_FUNC(
    csr_scale_row) (
    void *_scale_factor,
    csr_sparse_row_t * arow)
{
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    REAL_T *scale_factor = _scale_factor;
    const int NNZ = arow->NNZ_;
    const int inc = 1;
   
    C_BLAS(SCAL) (&NNZ, scale_factor, arow->vals_, &inc);   
#endif
}
/** Scale an csr matrix - result is a new matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \return A scale version of matrix A.
 */
bml_matrix_csr_t *TYPED_FUNC(
    bml_scale_csr_new) (
    void *_scale_factor,
    bml_matrix_csr_t * A)
{
    bml_matrix_csr_t *B = TYPED_FUNC(bml_copy_csr_new) (A);

    TYPED_FUNC(bml_scale_csr) (_scale_factor, A, B);

    return B;
}

/** Scale an csr matrix.
 *
 *  \ingroup scale_group
 *
 *  \param A The matrix to be scaled
 *  \param B Scaled version of matrix A
 */
void TYPED_FUNC(
    bml_scale_csr) (
    void *_scale_factor,
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B)
{
#ifdef NOBLAS
    LOG_ERROR("No BLAS library");
#else
    if (A != B)
    {
        TYPED_FUNC(bml_copy_csr) (A, B);
    }

    const int N = A->N_;
        
#pragma omp parallel for
    for(int i=0; i<N; i++)
    {
       TYPED_FUNC(csr_scale_row)(_scale_factor, B->data_[i]);
    } 
#endif
}

void TYPED_FUNC(
    bml_scale_inplace_csr) (
    void *_scale_factor,
    bml_matrix_csr_t * A)
{
    TYPED_FUNC(bml_scale_csr) (_scale_factor, A, A);
}
