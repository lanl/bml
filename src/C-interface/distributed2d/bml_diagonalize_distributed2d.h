#ifndef __BML_DIAGONALIZE_DISTRIBUTED2D_H
#define __BML_DIAGONALIZE_DISTRIBUTED2D_H

#include "bml_types_distributed2d.h"

void bml_diagonalize_distributed2d(
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_t * eigenvectors);

void bml_diagonalize_distributed2d_single_real(
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors);

void bml_diagonalize_distributed2d_double_real(
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors);

void bml_diagonalize_distributed2d_single_complex(
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors);

void bml_diagonalize_distributed2d_double_complex(
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors);

#ifdef BML_USE_ELPA
void bml_diagonalize_distributed2d_elpa_single_real(
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors);

void bml_diagonalize_distributed2d_elpa_double_real(
    bml_matrix_distributed2d_t * A,
    void *eigenvalues,
    bml_matrix_distributed2d_t * eigenvectors);
#endif

#endif
