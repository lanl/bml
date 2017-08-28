#ifndef __BML_IMPORT_DENSE_H
#define __BML_IMPORT_DENSE_H

#include "bml_types_dense.h"

bml_matrix_dense_t *bml_import_from_dense_dense(
    const bml_matrix_precision_t matrix_precision,
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const double threshold,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_dense_t *bml_import_from_dense_dense_single_real(
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_dense_t *bml_import_from_dense_dense_double_real(
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_dense_t *bml_import_from_dense_dense_single_complex(
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_dense_t *bml_import_from_dense_dense_double_complex(
    const bml_dense_order_t order,
    const int N,
    const void *A,
    const bml_distribution_mode_t distrib_mode);

void *bml_export_to_dense_dense(
    const bml_matrix_dense_t * A,
    const bml_dense_order_t order);

void *bml_export_to_dense_dense_single_real(
    const bml_matrix_dense_t * A,
    const bml_dense_order_t order);

void *bml_export_to_dense_dense_double_real(
    const bml_matrix_dense_t * A,
    const bml_dense_order_t order);

void *bml_export_to_dense_dense_single_complex(
    const bml_matrix_dense_t * A,
    const bml_dense_order_t order);

void *bml_export_to_dense_dense_double_complex(
    const bml_matrix_dense_t * A,
    const bml_dense_order_t order);

#endif
