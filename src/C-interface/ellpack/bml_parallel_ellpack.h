#ifndef __BML_PARALLEL_ELLPACK_H
#define __BML_PARALLEL_ELLPACK_H

#include "bml_types_ellpack.h"

void bml_allGatherVParallel_ellpack(
     bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_single_real(
     bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_double_real(
     bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_single_complex(
     bml_matrix_ellpack_t * A);

void bml_allGatherVParallel_ellpack_double_complex(
     bml_matrix_ellpack_t * A);

#endif
