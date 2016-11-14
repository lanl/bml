#ifndef __BML_DIAGONALIZE_ELLPACK_H
#define __BML_DIAGONALIZE_ELLPACK_H

#include "bml_types_ellpack.h"
#include "dense/bml_types_dense.h"

void bml_diagonalize_ellpack(
    const bml_matrix_ellpack_t * A,
    double *eigenvalues,
    bml_matrix_dense_t * eigenvectors);

void bml_diagonalize_ellpack_single_real(
    const bml_matrix_ellpack_t * A,
    double *eigenvalues,
    bml_matrix_dense_t * eigenvectors);

void bml_diagonalize_ellpack_double_real(
    const bml_matrix_ellpack_t * A,
    double *eigenvalues,
    bml_matrix_dense_t * eigenvectors);

void bml_diagonalize_ellpack_single_complex(
    const bml_matrix_ellpack_t * A,
    double *eigenvalues,
    bml_matrix_dense_t * eigenvectors);

void bml_diagonalize_ellpack_double_complex(
    const bml_matrix_ellpack_t * A,
    double *eigenvalues,
    bml_matrix_dense_t * eigenvectors);

#endif
