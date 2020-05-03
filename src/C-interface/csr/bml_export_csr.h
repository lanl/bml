#ifndef __BML_EXPORT_CSR_H
#define __BML_EXPORT_CSR_H

#include "bml_types_csr.h"

void *bml_export_to_dense_csr(
    bml_matrix_csr_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_csr_single_real(
    bml_matrix_csr_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_csr_double_real(
    bml_matrix_csr_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_csr_single_complex(
    bml_matrix_csr_t * A,
    bml_dense_order_t order);

void *bml_export_to_dense_csr_double_complex(
    bml_matrix_csr_t * A,
    bml_dense_order_t order);

#endif
