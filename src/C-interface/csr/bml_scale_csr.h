#ifndef __BML_SCALE_CSR_H
#define __BML_SCALE_CSR_H

#include "bml_types_csr.h"

#include <complex.h>

void csr_scale_row_single_real(
    void *_scale_factor,
    csr_sparse_row_t * arow);

void csr_scale_row_double_real(
    void *_scale_factor,
    csr_sparse_row_t * arow);

void csr_scale_row_single_complex(
    void *_scale_factor,
    csr_sparse_row_t * arow);

void csr_scale_row_double_complex(
    void *_scale_factor,
    csr_sparse_row_t * arow);

bml_matrix_csr_t *bml_scale_csr_new(
    void *scale_factor,
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_scale_csr_new_single_real(
    void *scale_factor,
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_scale_csr_new_double_real(
    void *scale_factor,
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_scale_csr_new_single_complex(
    void *scale_factor,
    bml_matrix_csr_t * A);

bml_matrix_csr_t *bml_scale_csr_new_double_complex(
    void *scale_factor,
    bml_matrix_csr_t * A);

void bml_scale_csr(
    void *scale_factor,
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_scale_csr_single_real(
    void *scale_factor,
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_scale_csr_double_real(
    void *scale_factor,
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_scale_csr_single_complex(
    void *scale_factor,
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_scale_csr_double_complex(
    void *scale_factor,
    bml_matrix_csr_t * A,
    bml_matrix_csr_t * B);

void bml_scale_inplace_csr(
    void *scale_factor,
    bml_matrix_csr_t * A);

void bml_scale_inplace_csr_single_real(
    void *scale_factor,
    bml_matrix_csr_t * A);

void bml_scale_inplace_csr_double_real(
    void *scale_factor,
    bml_matrix_csr_t * A);

void bml_scale_inplace_csr_single_complex(
    void *scale_factor,
    bml_matrix_csr_t * A);

void bml_scale_inplace_csr_double_complex(
    void *scale_factor,
    bml_matrix_csr_t * A);

#endif
