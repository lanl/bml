#ifndef __BML_EXPORT_DISTRIBUTED2D_H
#define __BML_EXPORT_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

void *bml_export_to_dense_distributed2d(
    bml_matrix_distributed2d_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_distributed2d_single_real(
    bml_matrix_distributed2d_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_distributed2d_double_real(
    bml_matrix_distributed2d_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_distributed2d_single_complex(
    bml_matrix_distributed2d_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_distributed2d_double_complex(
    bml_matrix_distributed2d_t * A,
    bml_dense_order_t order);

#endif
