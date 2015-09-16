/** \file */

#ifndef __BML_INTROSPECTION_H
#define __BML_INTROSPECTION_H

#include "bml_types.h"

bml_matrix_type_t bml_get_type(const bml_matrix_t *A);

int bml_get_size(const bml_matrix_t *A);
#endif
