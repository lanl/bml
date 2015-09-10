#ifndef __BML_ALLOCATE_H
#define __BML_ALLOCATE_H

#include "bml_types.h"

#include <stdlib.h>

void *bml_allocate_memory(const size_t s);

void bml_free_memory(void *ptr);

void bml_allocate(const bml_matrix_type_t matrix_type,
                  const bml_matrix_precision_t matrix_precision,
                  bml_matrix_t *A,
                  const int N);

void bml_deallocate(bml_matrix_t **A);

#endif
