#ifndef __BML_DIAGONALIZE_H
#define __BML_DIAGONALIZE_H

#include "bml_types.h"

void bml_diagonalize(
    const bml_matrix_t * A,
    double **eigenvalues,
    bml_matrix_t ** eigenvectors);

#endif
