#ifndef __BML_ADD_DENSE_H
#define __BML_ADD_DENSE_H

#include "bml_types_dense.h"

void bml_add_dense(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

void bml_add_dense_single_real(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

void bml_add_dense_double_real(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

void bml_add_dense_single_complex(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

void bml_add_dense_double_complex(
    const bml_matrix_dense_t * A,
    const bml_matrix_dense_t * B,
    const double alpha,
    const double beta);

void bml_add_identity_dense(
    const bml_matrix_dense_t * A,
    const double beta);

void bml_add_identity_dense_single_real(
    const bml_matrix_dense_t * A,
    const double beta);

void bml_add_identity_dense_double_real(
    const bml_matrix_dense_t * A,
    const double beta);

void bml_add_identity_dense_single_complex(
    const bml_matrix_dense_t * A,
    const double beta);

void bml_add_identity_dense_double_complex(
    const bml_matrix_dense_t * A,
    const double beta);

#endif
