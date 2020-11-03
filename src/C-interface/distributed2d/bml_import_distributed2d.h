#ifndef __BML_IMPORT_DISTRIBUTED2D_H
#define __BML_IMPORT_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

bml_matrix_distributed2d_t *bml_import_from_dense_distributed2d(
    bml_matrix_type_t matrix_type,
    bml_matrix_precision_t matrix_precision,
    bml_dense_order_t order,
    int N,
    void *A,
    double threshold,
    int M);

bml_matrix_distributed2d_t
    *
bml_import_from_dense_distributed2d_single_real(bml_matrix_type_t matrix_type,
                                                bml_dense_order_t order,
                                                int N,
                                                void *A,
                                                double threshold,
                                                int M);

bml_matrix_distributed2d_t
    *
bml_import_from_dense_distributed2d_double_real(bml_matrix_type_t matrix_type,
                                                bml_dense_order_t order,
                                                int N,
                                                void *A,
                                                double threshold,
                                                int M);

bml_matrix_distributed2d_t
    *
bml_import_from_dense_distributed2d_single_complex(bml_matrix_type_t
                                                   matrix_type,
                                                   bml_dense_order_t order,
                                                   int N,
                                                   void *A,
                                                   double threshold,
                                                   int M);

bml_matrix_distributed2d_t
    *
bml_import_from_dense_distributed2d_double_complex(bml_matrix_type_t
                                                   matrix_type,
                                                   bml_dense_order_t order,
                                                   int N,
                                                   void *A,
                                                   double threshold,
                                                   int M);

#endif
