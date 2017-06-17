#ifndef __BML_INVERSE_ELLPACK_H
#define __BML_INVERSE_ELLPACK_H

#include "bml_types_ellpack.h"

bml_matrix_ellpack_t *bml_inverse_ellpack(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_inverse_ellpack_single_real(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_inverse_ellpack_double_real(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_inverse_ellpack_single_complex(
    const bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_inverse_ellpack_double_complex(
    const bml_matrix_ellpack_t * A);

#endif
