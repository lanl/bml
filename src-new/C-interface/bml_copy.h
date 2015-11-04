/** \file */

#ifndef __BML_COPY_H
#    define __BML_COPY_H

#    include "bml_types.h"

bml_matrix_t *bml_copy_new(
    const bml_matrix_t * A);

void bml_copy(
    const bml_matrix_t * A,
    bml_matrix_t * B);

#endif
