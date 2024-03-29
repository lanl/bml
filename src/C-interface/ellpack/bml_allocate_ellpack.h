#ifndef __BML_ALLOCATE_ELLPACK_H
#define __BML_ALLOCATE_ELLPACK_H

#include "bml_types_ellpack.h"
#ifdef BML_USE_ROCSPARSE
// Copy rocsparse headers into src/rocsparse/ and edit rocsparse_functions.h to remove '[[...]]' text
#include "../rocsparse/rocsparse.h"
/* DEBUG
#include <hip/hip_runtime.h> // needed for hipDeviceSynchronize()
*/
#endif

void bml_deallocate_ellpack(
    bml_matrix_ellpack_t * A);

void bml_deallocate_ellpack_single_real(
    bml_matrix_ellpack_t * A);

void bml_deallocate_ellpack_double_real(
    bml_matrix_ellpack_t * A);

void bml_deallocate_ellpack_single_complex(
    bml_matrix_ellpack_t * A);

void bml_deallocate_ellpack_double_complex(
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

bml_matrix_ellpack_t *bml_noinit_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    bml_matrix_dimension_t matrix_dimension,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t
    * bml_noinit_matrix_ellpack_single_real(bml_matrix_dimension_t
                                            matrix_dimension,
                                            bml_distribution_mode_t
                                            distrib_mode);

bml_matrix_ellpack_t
    * bml_noinit_matrix_ellpack_double_real(bml_matrix_dimension_t
                                            matrix_dimension,
                                            bml_distribution_mode_t
                                            distrib_mode);

bml_matrix_ellpack_t
    * bml_noinit_matrix_ellpack_single_complex(bml_matrix_dimension_t
                                               matrix_dimension,
                                               bml_distribution_mode_t
                                               distrib_mode);

bml_matrix_ellpack_t
    * bml_noinit_matrix_ellpack_double_complex(bml_matrix_dimension_t
                                               matrix_dimension,
                                               bml_distribution_mode_t
                                               distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_single_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_double_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_single_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_zero_matrix_ellpack_double_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_single_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_double_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_single_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_banded_matrix_ellpack_double_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_single_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_double_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_single_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_random_matrix_ellpack_double_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_single_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_double_real(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_single_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_ellpack_t *bml_identity_matrix_ellpack_double_complex(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode);

void bml_update_domain_ellpack(
    bml_matrix_ellpack_t * A,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart);

#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE) || defined(BML_USE_HYPRE)
void bml_ellpack2cucsr_ellpack(
    bml_matrix_ellpack_t * A);
void bml_ellpack2cucsr_ellpack_single_real(
    bml_matrix_ellpack_t * A);
void bml_ellpack2cucsr_ellpack_double_real(
    bml_matrix_ellpack_t * A);
void bml_ellpack2cucsr_ellpack_single_complex(
    bml_matrix_ellpack_t * A);
void bml_ellpack2cucsr_ellpack_double_complex(
    bml_matrix_ellpack_t * A);

void bml_cucsr2ellpack_ellpack(
    bml_matrix_ellpack_t * A);
void bml_cucsr2ellpack_ellpack_single_real(
    bml_matrix_ellpack_t * A);
void bml_cucsr2ellpack_ellpack_double_real(
    bml_matrix_ellpack_t * A);
void bml_cucsr2ellpack_ellpack_single_complex(
    bml_matrix_ellpack_t * A);
void bml_cucsr2ellpack_ellpack_double_complex(
    bml_matrix_ellpack_t * A);
#endif

#ifdef BML_USE_ROCSPARSE
void bml_sort_rocsparse_ellpack(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A);
void bml_sort_rocsparse_ellpack_single_real(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A);
void bml_sort_rocsparse_ellpack_double_real(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A);
void bml_sort_rocsparse_ellpack_single_complex(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A);
void bml_sort_rocsparse_ellpack_double_complex(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A);
void bml_prune_rocsparse_ellpack(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A,
    double threshold);
void bml_prune_rocsparse_ellpack_single_real(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A,
    double threshold);
void bml_prune_rocsparse_ellpack_double_real(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A,
    double threshold);
void bml_prune_rocsparse_ellpack_single_complex(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A,
    double threshold);
void bml_prune_rocsparse_ellpack_double_complex(
    rocsparse_handle handle,
    bml_matrix_ellpack_t * A,
    double threshold);
#endif
#endif
