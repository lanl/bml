/** \file */

#ifndef __BML_SETTERS_H
#define __BML_SETTERS_H

#include "bml_types.h"

void bml_set_element_new(
  bml_matrix_t * A,
  const int i,
  const int j,
  const void *value);

void bml_set_element(
    bml_matrix_t * A,
    const int i,
    const int j,
    const void *value);

void bml_set_row(
    bml_matrix_t * A,
    const int i,
    const void *row,
    const double threshold);

void bml_set_diagonal(
    bml_matrix_t * A,
    const void *diagonal,
    const double threshold);

#endif
