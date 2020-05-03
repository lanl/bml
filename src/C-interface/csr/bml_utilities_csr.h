/** \file */

#ifndef __BML_UTILITIES_CSR_H
#define __BML_UTILITIES_CSR_H

#include "bml_types_csr.h"

#include <stdio.h>
#include <stdlib.h>

void bml_read_bml_matrix_csr(
    bml_matrix_csr_t * A,
    char *filename);

void bml_read_bml_matrix_csr_single_real(
    bml_matrix_csr_t * A,
    char *filename);

void bml_read_bml_matrix_csr_double_real(
    bml_matrix_csr_t * A,
    char *filename);

void bml_read_bml_matrix_csr_single_complex(
    bml_matrix_csr_t * A,
    char *filename);

void bml_read_bml_matrix_csr_double_complex(
    bml_matrix_csr_t * A,
    char *filename);

void bml_write_bml_matrix_csr(
    bml_matrix_csr_t * A,
    char *filename);

void bml_write_bml_matrix_csr_single_real(
    bml_matrix_csr_t * A,
    char *filename);

void bml_write_bml_matrix_csr_double_real(
    bml_matrix_csr_t * A,
    char *filename);

void bml_write_bml_matrix_csr_single_complex(
    bml_matrix_csr_t * A,
    char *filename);

void bml_write_bml_matrix_csr_double_complex(
    bml_matrix_csr_t * A,
    char *filename);

#endif
