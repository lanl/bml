/** \file */

#ifndef __BML_UTILITIES_ELLBLOCK_H
#define __BML_UTILITIES_ELLBLOCK_H

#include "bml_types_ellblock.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

void bml_read_bml_matrix_ellblock(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_read_bml_matrix_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_read_bml_matrix_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_read_bml_matrix_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_read_bml_matrix_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_write_bml_matrix_ellblock(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_write_bml_matrix_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_write_bml_matrix_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_write_bml_matrix_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const char *filename);

void bml_write_bml_matrix_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const char *filename);

double bml_sum_squares_single_real(
    const float *v,
    const int,
    const int,
    const int);
double bml_sum_squares_double_real(
    const double *v,
    const int,
    const int,
    const int);
double bml_sum_squares_single_complex(
    const float complex * v,
    const int,
    const int,
    const int);
double bml_sum_squares_double_complex(
    const double complex * v,
    const int,
    const int,
    const int);

double bml_norm_inf_single_real(
    const float *v,
    const int,
    const int,
    const int);
double bml_norm_inf_double_real(
    const double *v,
    const int,
    const int,
    const int);
double bml_norm_inf_single_complex(
    const float complex * v,
    const int,
    const int,
    const int);
double bml_norm_inf_double_complex(
    const double complex * v,
    const int,
    const int,
    const int);

double bml_norm_inf_fast_single_real(
    const float *v,
    const int);
double bml_norm_inf_fast_double_real(
    const double *v,
    const int);
double bml_norm_inf_fast_single_complex(
    const float complex * v,
    const int);
double bml_norm_inf_fast_double_complex(
    const double complex * v,
    const int);

#endif
