#ifndef __BML_TYPES_DENSE_H
#define __BML_TYPES_DENSE_H

#include "../bml_types.h"

/** Dense matrix type. */
struct bml_matrix_dense_t {
    /** The matrix type identifier. */
    bml_matrix_type_t matrix_type;
    /** The number of rows/columns. */
    int N;
    /** The dense matrix. */
    double *matrix;
};
typedef struct bml_matrix_dense_t bml_matrix_dense_t;

#endif
