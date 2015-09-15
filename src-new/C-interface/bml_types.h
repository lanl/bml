/** \file */

#ifndef __BML_TYPES_H
#define __BML_TYPES_H

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
    /** Matrix data is stored in single precision (float). */
    single_precision,
    /** Matrix data is stored in double precision (double). */
    double_precision
} bml_matrix_precision_t;

/** The matrix type. */
typedef void bml_matrix_t;

#endif
