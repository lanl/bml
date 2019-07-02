#ifndef __BML_ADD_ELLBLOCK_H
#define __BML_ADD_ELLBLOCK_H

#include "bml_types_ellblock.h"

void bml_add_ellblock(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellblock_single_real(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellblock_double_real(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellblock_single_complex(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_ellblock_double_complex(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellblock(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellblock_single_real(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellblock_double_real(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellblock_single_complex(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

double bml_add_norm_ellblock_double_complex(
    bml_matrix_ellblock_t * const A,
    bml_matrix_ellblock_t const *const B,
    double const alpha,
    double const beta,
    double const threshold);

void bml_add_identity_ellblock(
    bml_matrix_ellblock_t * const A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_single_real(
    bml_matrix_ellblock_t * const A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_double_real(
    bml_matrix_ellblock_t * const A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_single_complex(
    bml_matrix_ellblock_t * const A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_double_complex(
    bml_matrix_ellblock_t * const A,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock(
    bml_matrix_ellblock_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_single_real(
    bml_matrix_ellblock_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_double_real(
    bml_matrix_ellblock_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_single_complex(
    bml_matrix_ellblock_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_double_complex(
    bml_matrix_ellblock_t * const A,
    const double alpha,
    const double beta,
    const double threshold);

#endif
