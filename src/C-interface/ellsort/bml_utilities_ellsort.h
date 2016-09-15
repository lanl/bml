/** \file */

#ifndef __BML_UTILITIES_ELLSORT_H
#define __BML_UTILITIES_ELLSORT_H

#include "bml_types_ellsort.h"

#include <stdio.h>
#include <stdlib.h>

void bml_read_bml_matrix_ellsort(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_read_bml_matrix_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_read_bml_matrix_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_read_bml_matrix_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_read_bml_matrix_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_write_bml_matrix_ellsort(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_write_bml_matrix_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_write_bml_matrix_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_write_bml_matrix_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const char *filename);

void bml_write_bml_matrix_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const char *filename);

#endif
