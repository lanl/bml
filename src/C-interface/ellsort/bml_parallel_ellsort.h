#ifndef __BML_PARALLEL_ELLSORT_H
#define __BML_PARALLEL_ELLSORT_H

#include "bml_types_ellsort.h"

void bml_allGatherVParallel_ellsort(
     bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_single_real(
     bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_double_real(
     bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_single_complex(
     bml_matrix_ellsort_t * A);

void bml_allGatherVParallel_ellsort_double_complex(
     bml_matrix_ellsort_t * A);

#endif
