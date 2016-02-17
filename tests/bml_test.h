#ifndef __BML_TEST_H
#define __BML_TEST_H

#include <bml.h>

typedef int (
    *test_function_t) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M);

#include "add_matrix.h"
#include "allocate_matrix.h"
#include "convert_matrix.h"
#include "copy_matrix.h"
#include "diagonalize_matrix.h"
#include "multiply_matrix.h"
#include "normalize_matrix.h"

#endif
