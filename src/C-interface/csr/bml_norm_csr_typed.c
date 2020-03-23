#include "../../macros.h"
#include "../../typed.h"
#include "../bml_norm.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_norm_csr.h"
#include "bml_types_csr.h"
#include "bml_logger.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/** Calculate the sum of squares of the elements of a csr matrix.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \return The sum of squares of A
 */
double TYPED_FUNC(
    bml_sum_squares_csr) (
    bml_matrix_csr_t * A)
{
    int N = A->N_;

    REAL_T sum = 0.0;

#pragma omp parallel for                        \
  shared(N)                  \
  reduction(+:sum)
    for (int i = 0; i < N; i++)
    {
        REAL_T *vals = (REAL_T *)A->data_[i]->vals_;
        const int annz = A->data_[i]->NNZ_;        
        for (int pos = 0; pos < annz; pos++)
        {
            REAL_T xval = vals[pos];
            sum += xval * xval;
        }
    }

    return (double) REAL_PART(sum);
}

/** Calculate the sum of squares of a principal submatrix.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix
 *  \param core_pos Core rows of submatrix
 *  \param core_size Number of core rows
 *  \return The sum of squares of A
 */
double TYPED_FUNC(
    bml_sum_squares_submatrix_csr) (
    bml_matrix_csr_t * A,
    int core_size)
{
    REAL_T sum = 0.0;   

#pragma omp parallel for                        \
  shared(core_size)         \
  reduction(+:sum)
    for (int i = 0; i < core_size; i++)
    {
        int *cols = A->data_[i]->cols_;
        REAL_T *vals = (REAL_T *)A->data_[i]->vals_;
        const int annz = A->data_[i]->NNZ_;  
        for (int pos = 0; pos < annz; pos++)
        {
            if (cols[pos] < core_size)
            {
                REAL_T value = vals[pos];
                sum += value * value;
            }
        }
    }
    return (double) REAL_PART(sum);
}


/** Calculate the sum of squares of the elements of \alpha A + \beta B.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \param alpha Multiplier for A
 *  \param beta Multiplier for B
 *  \pram threshold Threshold
 *  \return The sum of squares of \alpha A + \beta B
 */
double TYPED_FUNC(
    bml_sum_squares2_csr) (
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B,
    double alpha,
    double beta,
    double threshold)
{
    LOG_ERROR("bml_sum_squares2_csr:  Not implemented");
    return 0.;
}

/** Calculate the Frobenius norm of matrix A.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \return The Frobenius norm of A
 */
double TYPED_FUNC(
    bml_fnorm_csr) (
    bml_matrix_csr_t * A)
{
    double fnorm = TYPED_FUNC(bml_sum_squares_csr) (A);
#ifdef DO_MPI
    if (bml_getNRanks() > 1 && A->distribution_mode == distributed)
    {
        bml_sumRealReduce(&fnorm);
    }
#endif
    fnorm = sqrt(fnorm);

    return (double) REAL_PART(fnorm);
}

/** Calculate the Frobenius norm of 2 matrices.
 *
 *  \ingroup norm_group
 *
 *  \param A The matrix A
 *  \param B The matrix B
 *  \return The Frobenius norm of A-B
 */
double TYPED_FUNC(
    bml_fnorm2_csr) (
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B)
{
    LOG_ERROR("bml_fnorm2_csr:  Not implemented");
    return 0.;
}
