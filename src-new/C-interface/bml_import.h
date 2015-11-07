/** \file */

#ifndef __BML_IMPORT_H
#define __BML_IMPORT_H

#include "bml_types.h"

bml_matrix_t *bml_import_from_dense(
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const void *A,
    const double threshold,
    const int M);

bml_matrix_t *bml_convert_from_dense(
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int N,
    const void *A,
    const double threshold,
    const int M);

#endif
