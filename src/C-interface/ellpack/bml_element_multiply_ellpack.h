#ifndef __BML_ELEMNET_MULTIPLY_ELLPACK_H
#define __BML_ELEMNET_MULTIPLY_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_element_multiply_AB_ellpack(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_element_multiply_AB_ellpack_single_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_element_multiply_AB_ellpack_double_real(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_element_multiply_AB_ellpack_single_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

void bml_element_multiply_AB_ellpack_double_complex(
    bml_matrix_ellpack_t * A,
    bml_matrix_ellpack_t * B,
    bml_matrix_ellpack_t * C,
    double threshold);

#endif
