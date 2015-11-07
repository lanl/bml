#ifndef __BML_DIAGONALIZE_DENSE_H
#define __BML_DIAGONALIZE_DENSE_H

#include "bml_types_dense.h"

void bml_diagonalize_dense(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_t ** eigenvectors);

void bml_diagonalize_dense_single_real(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors);

void bml_diagonalize_dense_double_real(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors);

void bml_diagonalize_dense_single_complex(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors);

void bml_diagonalize_dense_double_complex(
    const bml_matrix_dense_t * A,
    double **eigenvalues,
    bml_matrix_dense_t ** eigenvectors);

#endif
