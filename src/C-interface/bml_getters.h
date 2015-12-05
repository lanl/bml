/** \file */

#ifndef __BML_GETTERS_H
#define __BML_GETTERS_H

#include "bml_types.h"

/*
void bml_get(
    bml_matrix_t * A,
    const int i,
    const int j,
    void *value);
*/

void bml_get_row(
    bml_matrix_t * A,
    const int i,
    void *row);

#endif
