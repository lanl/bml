/** \file */

#ifndef __BML_SETTERS_H
#define __BML_SETTERS_H

#include "bml_types.h"

void bml_set(
    bml_matrix_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row(
    bml_matrix_t * A,
    const int i,
    const void *row,
    const double threshold);

#endif
