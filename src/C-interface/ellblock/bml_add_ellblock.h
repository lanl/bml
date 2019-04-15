#ifndef __BML_ADD_ELLBLOCK_H
#define __BML_ADD_ELLBLOCK_H

#include "bml_types_ellblock.h"

void bml_add_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

void bml_add_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

void bml_add_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

void bml_add_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

void bml_add_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_add_norm_ellblock(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_add_norm_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_add_norm_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_add_norm_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

double bml_add_norm_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const bml_matrix_ellblock_t * B,
    const double alpha,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock(
    const bml_matrix_ellblock_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const double beta,
    const double threshold);

void bml_add_identity_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock(
    const bml_matrix_ellblock_t * A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_single_real(
    const bml_matrix_ellblock_t * A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_double_real(
    const bml_matrix_ellblock_t * A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_single_complex(
    const bml_matrix_ellblock_t * A,
    const double alpha,
    const double beta,
    const double threshold);

void bml_scale_add_identity_ellblock_double_complex(
    const bml_matrix_ellblock_t * A,
    const double alpha,
    const double beta,
    const double threshold);

#endif
