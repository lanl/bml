#include "bml_types.h"
#include "bml_logger.h"

#include <stdio.h>

/** Print a dense matrix.
 *
 * \param N The number of rows/columns.
 * \param i_l The lower row index.
 * \param i_u The upper row index.
 * \param j_l The lower column index.
 * \param j_u The upper column index.
 * \param matrix_precision The real precision.
 * \param A The matrix.
 */
void bml_print_matrix(const int N,
                      const int i_l, const int i_u,
                      const int j_l, const int j_u,
                      bml_matrix_precision_t matrix_precision,
                      void *A)
{
    float *A_float;
    double *A_double;

    switch(matrix_precision) {
    case single_precision:
    {
        A_float = A;
        for(int i = i_l; i < i_u; i++) {
            for(int j = j_l; j < j_u; j++) {
                printf("% 1.3f", A_float[i+j*N]);
            }
            printf("\n");
        }
        break;
    }
    case double_precision:
    {
        A_double = A;
        for(int i = i_l; i < i_u; i++) {
            for(int j = j_l; j < j_u; j++) {
                printf("% 1.3f", A_double[i+j*N]);
            }
            printf("\n");
        }
        break;
    }
    default:
        LOG_ERROR("unknown matrix precision\n");
        break;
    }
}
