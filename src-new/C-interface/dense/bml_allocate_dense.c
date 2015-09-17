#include "bml_allocate.h"
#include "bml_allocate_dense.h"
#include "bml_types.h"
#include "bml_types_dense.h"

/** Deallocate a matrix.
 *
 * \ingroup allocate_group
 *
 * \param A The matrix.
 */
void bml_deallocate_dense(bml_matrix_dense_t *A)
{
    bml_free_memory(A->matrix);
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
 *  \return The matrix.
 */
bml_matrix_dense_t *bml_zero_matrix_dense(const bml_matrix_precision_t matrix_precision,
                                          const int N)
{
    bml_matrix_dense_t *A = NULL;

    A = bml_allocate_memory(sizeof(bml_matrix_dense_t));
    A->matrix_type = dense;
    A->matrix_precision = matrix_precision;
    A->N = N;
    switch(matrix_precision) {
    case single_precision:
        A->matrix = bml_allocate_memory(sizeof(float)*N*N);
        break;
    case double_precision:
        A->matrix = bml_allocate_memory(sizeof(double)*N*N);
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
 *  \return The matrix.
 */
bml_matrix_dense_t *bml_random_matrix_dense(const bml_matrix_precision_t matrix_precision,
                                            const int N)
{
    bml_matrix_dense_t *A = bml_zero_matrix_dense(matrix_precision, N);
    float *A_float = NULL;
    double *A_double = NULL;

    switch(matrix_precision) {
    case single_precision:
        A_float = A->matrix;
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                A_float[i+N*j] = rand()/(float) RAND_MAX;
            }
        }
        break;
    case double_precision:
        A_double = A->matrix;
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                A_double[i+N*j] = rand()/(double) RAND_MAX;
            }
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
 *  \return The matrix.
 */
bml_matrix_dense_t *bml_identity_matrix_dense(const bml_matrix_precision_t matrix_precision,
                                              const int N)
{
    bml_matrix_dense_t *A = bml_zero_matrix_dense(matrix_precision, N);
    float *A_float = NULL;
    double *A_double = NULL;

    switch(matrix_precision) {
    case single_precision:
        A_float = A->matrix;
        for(int i = 0; i < N; i++) {
            A_float[i+N*i] = 1;
        }
        break;
    case double_precision:
        A_double = A->matrix;
        for(int i = 0; i < N; i++) {
            A_double[i+N*i] = 1;
        }
    }
    return A;
}
