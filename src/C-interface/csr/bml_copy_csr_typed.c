#include "../../macros.h"
#include "../../typed.h"
#include "../bml_logger.h"
#include "../bml_allocate.h"
#include "bml_allocate_csr.h"
#include "../bml_copy.h"
#include "../bml_types.h"
#include "bml_copy_csr.h"
#include "bml_types_csr.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/** Copy a csr matrix row - result is a new row.
 *
 *  \ingroup copy_group
 *
 *  \param arow The row to be copied
 *  \return A copy of arow.
 */
csr_sparse_row_t *TYPED_FUNC(
    csr_copy_row_new) (
    const csr_sparse_row_t * arow)
{
    const int alloc_size = arow->alloc_size_;
    csr_sparse_row_t *brow = TYPED_FUNC(csr_noinit_row) (alloc_size);

    const int NNZ = arow->NNZ_;
    brow->NNZ_ = NNZ;

    memcpy(brow->cols_, arow->cols_, NNZ * sizeof(int));
    memcpy(brow->vals_, arow->vals_, NNZ * sizeof(REAL_T));

    return brow;
}

/** Copy a csr matrix - result is a new matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \return A copy of matrix A.
 */
bml_matrix_csr_t *TYPED_FUNC(
    bml_copy_csr_new) (
    bml_matrix_csr_t * A)
{
    const int N = A->N_;
    bml_matrix_csr_t *B =
        bml_noinit_allocate_memory(sizeof(bml_matrix_csr_t));

    B->matrix_type = A->matrix_type;
    B->matrix_precision = A->matrix_precision;
    B->N_ = N;
    B->NZMAX_ = A->NZMAX_;
    B->TOTNNZ_ = A->TOTNNZ_;
    B->distribution_mode = A->distribution_mode;

    /** allocate csr row data */
    B->data_ = bml_noinit_allocate_memory(sizeof(csr_sparse_row_t *) * N);
    // copy rows
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        csr_sparse_row_t *new_row =
            TYPED_FUNC(csr_copy_row_new) ((A->data_)[i]);
        (B->data_)[i] = new_row;
    }

    // copy domain info
//    bml_copy_domain(A->domain, B->domain);
//    bml_copy_domain(A->domain2, B->domain2);

    return B;
}

/** Copy a csr matrix row.
 *
 *  \ingroup copy_group
 *
 *  \param arow The row to be copied
 *  \param brow Copy of arow. Assumes brow is already allocated
 *              and has the same size and number of entries as A.
 */
void TYPED_FUNC(
    csr_copy_row) (
    const csr_sparse_row_t * arow,
    csr_sparse_row_t * brow)
{
    const int NNZ = arow->NNZ_;
    // Check size for data
    if (brow->alloc_size_ < NNZ)
    {
        LOG_ERROR("CSR row size mismatch\n");
    }

    memcpy(brow->cols_, arow->cols_, NNZ * sizeof(int));
    memcpy(brow->vals_, arow->vals_, NNZ * sizeof(REAL_T));

    brow->NNZ_ = NNZ;
}

/** Copy a csr matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be copied
 *  \param B Copy of matrix A. Assumes B is already allocated
 *           and has the same size and number of entries as A.
 */
void TYPED_FUNC(
    bml_copy_csr) (
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B)
{
    const int N = A->N_;
    // check that sizes match
    assert(A->N_ == B->N_);
    assert(B->matrix_type == A->matrix_type);
    assert(B->matrix_precision == A->matrix_precision);

    B->TOTNNZ_ = A->TOTNNZ_;
    // copy rows
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        TYPED_FUNC(csr_copy_row) ((A->data_)[i], (B->data_)[i]);
    }
/*
    if (A->distribution_mode == B->distribution_mode)
    {
        bml_copy_domain(A->domain, B->domain);
        bml_copy_domain(A->domain2, B->domain2);
    }
*/
}



/** Reorder an csr matrix.
 *
 *  \ingroup copy_group
 *
 *  \param A The matrix to be reordered
 *  \param B The permutation vector
 */
void TYPED_FUNC(
    bml_reorder_csr) (
    bml_matrix_csr_t * A,
    int *perm)
{
    LOG_ERROR("bml_reorder_csr not implemented\n");
}
