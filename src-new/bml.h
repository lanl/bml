#ifndef __BML_H
#define __BML_H

typedef struct bml_matrix_t {
    void *ptr;
} bml_matrix_t;

typedef enum bml_matrix_type_t {
    dense
} bml_matrix_type_t;

typedef enum bml_matrix_precision_t {
    single_precision,
    double_precision
} bml_matrix_precision_t;

#include "bml_logger.h"
#include "bml_allocate.h"

#endif
