/** \file */

#ifndef __BML_EXPORT_H
#define __BML_EXPORT_H

#include "bml_types.h"

void *bml_convert_to_dense(
    const bml_matrix_t * A);

void *bml_export_to_dense(
    const bml_matrix_t * A);

#endif
