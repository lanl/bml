/** \file */

#ifndef __BML_EXPORT_H
#define __BML_EXPORT_H

#include "bml_types.h"

void *bml_export_to_dense(
    const bml_matrix_t * A,
    const bml_dense_order_t order);

#endif
