#ifndef __BML_ALLOCATE_ELLBLOCK_H
#define __BML_ALLOCATE_ELLBLOCK_H

#include "bml_types_ellblock.h"

void bml_deallocate_ellblock(
    bml_matrix_ellblock_t * A);

void bml_clear_ellblock(
    bml_matrix_ellblock_t * A);

void bml_clear_ellblock_single_real(
    bml_matrix_ellblock_t * A);

void bml_clear_ellblock_double_real(
    bml_matrix_ellblock_t * A);

void bml_clear_ellblock_single_complex(
    bml_matrix_ellblock_t * A);

void bml_clear_ellblock_double_complex(
    bml_matrix_ellblock_t * A);

bml_matrix_ellblock_t *bml_noinit_matrix_ellblock(
    const bml_matrix_precision_t matrix_precision,
    const bml_matrix_dimension_t matrix_dimension,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_noinit_matrix_ellblock_single_real(
    const bml_matrix_dimension_t matrix_dimension,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_noinit_matrix_ellblock_double_real(
    const bml_matrix_dimension_t matrix_dimension,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_noinit_matrix_ellblock_single_complex(
    const bml_matrix_dimension_t matrix_dimension,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_noinit_matrix_ellblock_double_complex(
    const bml_matrix_dimension_t matrix_dimension,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_zero_matrix_ellblock(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_zero_matrix_ellblock_single_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_zero_matrix_ellblock_double_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_zero_matrix_ellblock_single_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_zero_matrix_ellblock_double_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_random_matrix_ellblock(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_random_matrix_ellblock_single_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_random_matrix_ellblock_double_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_random_matrix_ellblock_single_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_random_matrix_ellblock_double_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_identity_matrix_ellblock(
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_identity_matrix_ellblock_single_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_identity_matrix_ellblock_double_real(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_identity_matrix_ellblock_single_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_identity_matrix_ellblock_double_complex(
    const int N,
    const int M,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_block_matrix_ellblock(
    const bml_matrix_precision_t matrix_precision,
    const int NB,
    const int MB,
    const int *bsizes,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_block_matrix_ellblock_single_real(
    const int NB,
    const int MB,
    const int *bsizes,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_block_matrix_ellblock_double_real(
    const int NB,
    const int MB,
    const int *bsizes,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_block_matrix_ellblock_single_complex(
    const int NB,
    const int MB,
    const int *bsizes,
    const bml_distribution_mode_t distrib_mode);

bml_matrix_ellblock_t *bml_block_matrix_ellblock_double_complex(
    const int NB,
    const int MB,
    const int *bsizes,
    const bml_distribution_mode_t distrib_mode);

void bml_update_domain_ellblock(
    bml_matrix_ellblock_t * A,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart);

void bml_set_block_sizes(
    const int *bsize,
    int NB,
    int MB);
int *bml_get_block_sizes(
    const int N,
    const int M);
int bml_get_mb(
    );
int bml_get_nb(
    );
int count_nelements(
    bml_matrix_ellblock_t * A);
#endif
