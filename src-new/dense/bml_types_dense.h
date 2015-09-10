#ifndef __BML_TYPES_DENSE_H
#define __BML_TYPES_DENSE_H

#include "../bml_types.h"

/** Dense matrix type. */
typedef struct bml_matrix_dense_t {
    /** The matrix type identifier. */
    bml_matrix_type_t matrix_type;
    /** The dense matrix. */
    double *matrix;
} bml_matrix_dense_t;

#endif
