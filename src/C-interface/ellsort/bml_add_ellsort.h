#ifndef __BML_ADD_ELLSORT_H
#define __BML_ADD_ELLSORT_H

#include "bml_types_ellsort.h"

void bml_add_ellsort(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellsort_single_real(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellsort_double_real(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellsort_single_complex(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellsort_double_complex(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellsort(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellsort_single_real(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellsort_double_real(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellsort_single_complex(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellsort_double_complex(
    bml_matrix_ellsort_t * const A,
    bml_matrix_ellsort_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_identity_ellsort(
    const bml_matrix_ellsort_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellsort_single_real(
    const bml_matrix_ellsort_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellsort_double_real(
    const bml_matrix_ellsort_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellsort_single_complex(
    const bml_matrix_ellsort_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellsort_double_complex(
    const bml_matrix_ellsort_t * A,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellsort(
    bml_matrix_ellsort_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellsort_single_real(
    bml_matrix_ellsort_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellsort_double_real(
    bml_matrix_ellsort_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellsort_single_complex(
    bml_matrix_ellsort_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellsort_double_complex(
    bml_matrix_ellsort_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

#endif
