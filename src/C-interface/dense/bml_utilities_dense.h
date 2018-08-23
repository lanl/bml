/** \file */

#ifndef __BML_UTILITIES_DENSE_H
#define __BML_UTILITIES_DENSE_H

#include "bml_types_dense.h"

#include <stdio.h>
#include <stdlib.h>

void bml_print_bml_matrix_dense(
    const bml_matrix_dense_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

void bml_print_bml_matrix_dense_single_real(
    const bml_matrix_dense_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

void bml_print_bml_matrix_dense_double_real(
    const bml_matrix_dense_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

void bml_print_bml_matrix_dense_single_complex(
    const bml_matrix_dense_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

void bml_print_bml_matrix_dense_double_complex(
    const bml_matrix_dense_t * A,
    const int i_l,
    const int i_u,
    const int j_l,
    const int j_u);

void bml_read_bml_matrix_dense(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_read_bml_matrix_dense_single_real(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_read_bml_matrix_dense_double_real(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_read_bml_matrix_dense_single_complex(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_read_bml_matrix_dense_double_complex(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_write_bml_matrix_dense(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_write_bml_matrix_dense_single_real(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_write_bml_matrix_dense_double_real(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_write_bml_matrix_dense_single_complex(
    const bml_matrix_dense_t * A,
    const char *filename);

void bml_write_bml_matrix_dense_double_complex(
    const bml_matrix_dense_t * A,
    const char *filename);

#endif
