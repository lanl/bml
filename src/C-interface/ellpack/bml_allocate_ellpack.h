#ifndef __BML_ALLOCATE_ELLPACK_H
#define __BML_ALLOCATE_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_deallocate_ellpack(
    bml_matrix_ellpack_t * A);

void bml_clear_ellpack(
    bml_matrix_ellpack_t * A);

void bml_clear_ellpack_single_real(
    bml_matrix_ellpack_t * A);

void bml_clear_ellpack_double_real(
    bml_matrix_ellpack_t * A);

void bml_clear_ellpack_single_complex(
    bml_matrix_ellpack_t * A);

void bml_clear_ellpack_double_complex(
    bml_matrix_ellpack_t * A);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_single_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_double_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_single_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_double_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_single_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_double_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_single_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_double_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_single_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_double_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_single_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_double_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_single_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_double_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_single_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_double_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

void bml_update_domain_ellpack(
    bml_matrix_ellpack_t * A,
    int * localPartMin,
    int * localPartMax,
    int * nnodesInPart);

#endif
