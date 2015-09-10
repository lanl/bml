#ifndef __BML_TYPES_H
#define __BML_TYPES_H

/** The matrix type. */
typedef void bml_matrix_t;

/** The supported matrix types. */
typedef enum bml_matrix_type_t {
    dense,
    ellpack,
    csr
} bml_matrix_type_t;

/** The supported real precisions. */
typedef enum bml_matrix_precision_t {
    single_precision,
    double_precision
} bml_matrix_precision_t;

#endif
