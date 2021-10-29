#ifndef __BML_MULTIPLY_ELLPACK_H
#define __BML_MULTIPLY_ELLPACK_H

#include "bml_types_ellpack.h"
#include "stdlib.h"

void bml_multiply_ellpack(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha,
    double beta,
    double threshold);

void bml_multiply_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha,
    double beta,
    double threshold);

void bml_multiply_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha,
    double beta,
    double threshold);

void bml_multiply_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha,
    double beta,
    double threshold);

void bml_multiply_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double alpha,
    double beta,
    double threshold);

void *bml_multiply_x2_ellpack(
    bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    double threshold);

void *bml_multiply_x2_ellpack_single_real(
    bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    double threshold);

void *bml_multiply_x2_ellpack_double_real(
    bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    double threshold);

void *bml_multiply_x2_ellpack_single_complex(
    bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    double threshold);

void *bml_multiply_x2_ellpack_double_complex(
    bml_matrix_ellpack_t * X,
    bml_matrix_ellpack_t * X2,
    double threshold);

void bml_multiply_AB_ellpack(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_AB_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_AB_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_AB_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_AB_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_adjust_AB_ellpack(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_adjust_AB_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_adjust_AB_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_adjust_AB_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_multiply_adjust_AB_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);


#if defined(BML_USE_CUSPARSE)
#define BML_CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#endif
#endif
