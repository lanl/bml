#ifndef __BML_TYPES_H
#define __BML_TYPES_H

/** The matrix type. */
typedef void bml_matrix_t;

/** The supported matrix types. */
typedef enum {
    /** The matrix is not initialized. */
    uninitialized,
    /** Dense matrix. */
    dense,
    /** ELLPACK matrix. */
    ellpack,
    /** CSR matrix. */
    csr
} bml_matrix_type_t;

/** The supported real precisions. */
typedef enum {
    single_precision,
    double_precision
} bml_matrix_precision_t;

#endif
