/** \file */

#ifndef __BML_TYPES_H
#define __BML_TYPES_H

/** The supported matrix types. */
typedef enum
{
    /** The matrix is not initialized. */
    type_uninitialized,
    /** Dense matrix. */
    dense,
    /** ELLPACK matrix. */
    ellpack,
    /** BLOCK ELLPACK matrix. */
    ellblock,
    /** ELLSORT matrix. */
    ellsort,
    /** CSR matrix. */
    csr,
    /** distributed matrix. */
    distributed2d
} bml_matrix_type_t;

/** The supported real precisions. */
typedef enum
{
    /** The matrix is not initialized. */
    precision_uninitialized,
    /** Matrix data is stored in single precision (float). */
    single_real,
    /** Matrix data is stored in double precision (double). */
    double_real,
    /** Matrix data is stored in single-complex precision (float). */
    single_complex,
    /** Matrix data is stored in double-complex precision (double). */
    double_complex
} bml_matrix_precision_t;

/** The supported dense matrix elements orderings. */
typedef enum
{
    /** row-major order. */
    dense_row_major,
    /** column-major order. */
    dense_column_major
} bml_dense_order_t;

/** The vector type. */
typedef void bml_vector_t;

/** The matrix type. */
typedef void bml_matrix_t;

/** The matrix dimensions. */
typedef struct
{
    /** The number of rows. */
    int N_rows;
    /** The number of columns. */
    int N_cols;
    /** The maximum number of non-zeros per row (for ellpack). */
    int N_nz_max;
    /** The block sizes (for block_ellpack). */
    int *bsizes;
    /** The number of blocks/row (or column). */
    int NB;
} bml_matrix_dimension_t;

/** The supported distribution modes. */
typedef enum
{
    /** Each rank works on the full matrix. */
    sequential,
    /** Each rank works on its part of the matrix. */
    distributed,
    /** Each rank works on its set of graph partitions. */
    graph_distributed
} bml_distribution_mode_t;

/** Decomposition for working in parallel. */
struct bml_domain_t
{
   /** number of processors */
    int totalProcs;
   /** total number of rows */
    int totalRows;
   /** total number of columns */
    int totalCols;

   /** global minimum row number */
    int globalRowMin;
   /** global maximum row number */
    int globalRowMax;
   /** global total rows */
    int globalRowExtent;

   /** maximum extent for most processors */
    int maxLocalExtent;
   /** minimum extent for last processors */
    int minLocalExtent;
   /** minimum row per rank */
    int *localRowMin;
   /** maximum row per rank */
    int *localRowMax;
   /** extent of rows per rank, localRowMax - localRowMin */
    int *localRowExtent;
   /** local number of elements per rank */
    int *localElements;
   /** local displacements per rank for 2D */
    int *localDispl;
};
typedef struct bml_domain_t bml_domain_t;

#endif
