#include "../macros.h"
#include "../typed.h"
#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>

/** Allocate the zero matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_zero_matrix_ellpack) (
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A =
        bml_allocate_memory(sizeof(bml_matrix_ellpack_t));
    A->matrix_type = ellpack;
    A->matrix_precision = MATRIX_PRECISION;
    A->N = N;
    A->M = M;
    A->index = bml_allocate_memory(sizeof(int) * N * M);
    A->nnz = bml_allocate_memory(sizeof(int) * N);
    A->value = bml_allocate_memory(sizeof(REAL_T) * N * M);

    return A;
}

/** Allocate a random matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_random_matrix_ellpack) (
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = TYPED_FUNC(bml_zero_matrix_ellpack) (N, M);

    REAL_T *A_value = A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

    #pragma omp parallel for shared(A_value,A_index,A_nnz)
    for (int i = 0; i < N; i++)
    {
        int jind = 1;
        for (int j = 0; j < M; j++)
        {
            REAL_T rvalue = rand() / (REAL_T) RAND_MAX;
            
            if (i == j)
            {
                A_value[ROWMAJOR(i, 0, M)] = rvalue;
                A_index[ROWMAJOR(i, 0, M)] = j;
            }
            else
            {
                A_value[ROWMAJOR(i, jind, M)] = rvalue;
                A_index[ROWMAJOR(i, jind, M)] = j;
                jind++;
            }
        }
        A_nnz[i] = jind;
    }
    return A;
}

/** Allocate the identity matrix.
 *
 *  Note that the matrix \f$ a \f$ will be newly allocated. If it is
 *  already allocated then the matrix will be deallocated in the
 *  process.
 *
 *  \ingroup allocate_group
 *
 *  \param matrix_precision The precision of the matrix. The default
 *  is double precision.
 *  \param N The matrix size.
 *  \param M The number of non-zeroes per row.
 *  \return The matrix.
 */
bml_matrix_ellpack_t *TYPED_FUNC(
    bml_identity_matrix_ellpack) (
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = TYPED_FUNC(bml_zero_matrix_ellpack) (N, M);

    REAL_T *A_value = A->value;
    int *A_index = A->index;
    int *A_nnz = A->nnz;

    #pragma omp parallel for default(none) shared(A_value,A_index,A_nnz)
    for (int i = 0; i < N; i++)
    {
        A_value[ROWMAJOR(i, 0, M)] = (REAL_T) 1.0;
        A_index[ROWMAJOR(i, 0, M)] = i;
        A_nnz[i] = 1;
    }
    return A;
}
