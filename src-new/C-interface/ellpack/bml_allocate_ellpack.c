#include "bml_allocate.h"
#include "bml_allocate_ellpack.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <math.h>

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void
bml_deallocate_ellpack(
    bml_matrix_ellpack_t * A)
{
    bml_free_memory(A->value);
    bml_free_memory(A->index);
    bml_free_memory(A->nnz);
    bml_free_memory(A);
}

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
bml_matrix_ellpack_t *
bml_zero_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = NULL;

    A = bml_allocate_memory(sizeof(bml_matrix_ellpack_t));
    A->matrix_type = ellpack;
    A->matrix_precision = matrix_precision;
    A->N = N;
    A->M = M;
    A->index = bml_allocate_memory(sizeof(int) * N * M);
    A->nnz = bml_allocate_memory(sizeof(int) * N);

    // No values
    for (int i = 0; i < N; i++)
    {
        A->nnz[i] = 0;
    }

    switch (matrix_precision)
    {
    case single_real:
        A->value = bml_allocate_memory(sizeof(float) * N * M);
        break;
    case double_real:
        A->value = bml_allocate_memory(sizeof(double) * N * M);
        break;
    }
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
bml_matrix_ellpack_t *
bml_random_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = bml_zero_matrix_ellpack(matrix_precision, N, M);
    float *A_float = NULL;
    double *A_double = NULL;

    int *A_index = A->index;
    int *A_nnz = A->nnz;

    switch (matrix_precision)
    {
    case single_real:
        A_float = A->value;
        for (int i = 0; i < N; i++)
        {
            int jind = 0;
            for (int j = 0; j < M; j++)
            {
                float fvalue = rand() / (float) RAND_MAX;
                if (fabs(fvalue) > (float) 0.0)
                {
                    A_float[i * M + jind] = fvalue;
                    A_index[i * M + jind] = j;
                    jind++;
                }
            }
            A_nnz[i] = jind;
        }
        break;
    case double_real:
        A_double = A->value;
        for (int i = 0; i < N; i++)
        {
            int jind = 0;
            for (int j = 0; j < M; j++)
            {
                double dvalue = rand() / (double) RAND_MAX;
                if (fabs(dvalue) > (double) 0.0)
                {
                    A_double[i * M + jind] = dvalue;
                    A_index[i * M + jind] = j;
                    jind++;
                }
            }
            A_nnz[i] = jind;
        }
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
bml_matrix_ellpack_t *
bml_identity_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M)
{
    bml_matrix_ellpack_t *A = bml_zero_matrix_ellpack(matrix_precision, N, M);
    float *A_float = NULL;
    double *A_double = NULL;

    int *A_index = A->index;
    int *A_nnz = A->nnz;

    switch (matrix_precision)
    {
    case single_real:
        A_float = A->value;
        for (int i = 0; i < N; i++)
        {
            A_float[i * M] = 1;
            A_index[i * M] = i;
            A_nnz[i] = 1;
        }
        break;
    case double_real:
        A_double = A->value;
        for (int i = 0; i < N; i++)
        {
            A_double[i * M] = 1;
            A_index[i * M] = i;
            A_nnz[i] = 1;
        }
    }
    return A;
}
