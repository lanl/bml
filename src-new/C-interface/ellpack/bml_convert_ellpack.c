#include "../bml_types.h"
#include "../bml_allocate.h"
#include "../bml_logger.h"
#include "bml_allocate_ellpack.h"
#include "bml_convert_ellpack.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    float *float_A = NULL;
    double *double_A = NULL;

    float *float_value = NULL;
    double *double_value = NULL;

    bml_matrix_ellpack_t *A_bml = bml_zero_matrix_ellpack(matrix_precision, N, M);

    switch(matrix_precision) {
    case single_precision:
        float_A = (float*)A;
        float_value = A_bml->value;
        for (int i = 0; i < N; i++) {
            int jind = 0;
            for (int j = 0; j < N; j++) {
                if (fabs(float_A[i*N+j]) > (float)threshold) {
                    float_value[i*M+jind] = float_A[i*N+j];
                    A_bml->index[i*M+jind] = j;
                    jind++;
                }
            }
            A_bml->nnz[i] = jind;
        }
        break;
    case double_precision:
        double_A = (double*)A;
        double_value = A_bml->value;
        for (int i = 0; i < N; i++) {
            int jind = 0;
            for (int j = 0; j < N; j++) {
                if (fabs(double_A[i*N+j]) > (double)0.0) {
                    double_value[i*M+jind] = double_A[i*N+j];
                    A_bml->index[i*M+jind] = j;
                    jind++;
                }
            }
            A_bml->nnz[i] = jind;
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

    float *float_value = NULL;
    double *double_value = NULL;
    int N = A->N;
    int M = A->M;

    switch(A->matrix_precision) {
    case single_precision:
        A_float = bml_allocate_memory(sizeof(float)*N*N);
        float_value = A->value;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < A->nnz[i]; j++) {
                A_float[i*N+A->index[i*M+j]] = float_value[i*M+j];
            }
        }
        return A_float;
        break;
    case double_precision:
        A_double = bml_allocate_memory(sizeof(double)*N*N);
        double_value = A->value;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < A->nnz[i]; j++) {
                A_double[i*N+A->index[i*M+j]] = double_value[i*M+j];
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
