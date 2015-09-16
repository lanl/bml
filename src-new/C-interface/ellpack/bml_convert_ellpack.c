#include "../bml_types.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "bml_allocate_ellpack.h"
#include "bml_convert_ellpack.h"

#include <stdlib.h>
#include <string.h>

/** Convert a dense matrix into a bml matrix.
 *
 * \ingroup convert_group
 *
 * \param N The number of rows/columns
 * \param matrix_precision The real precision
 * \param A The dense matrix
 * \param threshold The matrix element magnited threshold
 * \return The bml matrix
 */
bml_matrix_ellpack_t *bml_convert_from_dense_ellpack(const bml_matrix_precision_t matrix_precision,
                                                 const int N,
                                                 const void *A,
                                                 const double threshold,
                                                 const int M)
{
    bml_matrix_ellpack_t *A_bml = bml_zero_matrix_ellpack(matrix_precision, N);

    switch(matrix_precision) {
    case single_precision:
        for (int i = 0; i < N) {
            int jind = 0;
            for (int j = 0; j < N) {
                if (fabs(A[i*N+j]) > (float)0.0) {
                    A_bml->value[i*N+jind] = A[i*N+j];
                    A_bml->index[i*N+jind] = j;
                    jind++;
                }
            }
        }
        break;
    case double_precision:
        for (int i = 0; i < N) {
            int jind = 0;
            for (int j = 0; j < N) {
                if (fabs(A[i*N+j]) > (double)0.0) {
                    A_bml->value[i*N+jind] = A[i*N+j];
                    A_bml->index[i*N+jind] = j;
                    jind++;
                }
            }
        }
        break;
    default:
        LOG_ERROR("unknown precision\n");
        break;
    }
    return A_bml;
}

/** Convert a bml matrix into a dense matrix.
 *
 * \ingroup convert_group
 *
 * \param A The bml matrix
 * \return The dense matrix
 */
void *bml_convert_to_dense_ellpack(const bml_matrix_ellpack_t *A)
{
    float *A_float = NULL;
    double *A_double = NULL;

    switch(A->matrix_precision) {
    case single_precision:
        A_float = bml_allocate_memory(sizeof(float)*A->N*A->N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < A->nnz[i]; j++) {
                A_float[i*N+A->index[i*N+j]] = A->value[i*N+j];
            }
        }
        return A_float;
        break;
    case double_precision:
        A_double = bml_allocate_memory(sizeof(double)*A->N*A->N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < A->nnz[i]; j++) {
                A_double[i*N+A->index[i*N+j]] = A->value[i*N+j];
            }
        }
        return A_double;
        break;
    default:
        LOG_ERROR("unknown precision\n");
        break;
    }
    return NULL;
}
