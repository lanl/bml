#include "bml_utilities.h"

void bml_print_matrix_single(const int N,
                             const float *A,
                             const int i_l,
                             const int i_u,
                             const int j_l,
                             const int j_u)
{
    bml_print_matrix(N, single_real, A, i_l, i_u, j_l, j_u);
}

void bml_print_matrix_double(const int N,
                             const double *A,
                             const int i_l,
                             const int i_u,
                             const int j_l,
                             const int j_u)
{
    bml_print_matrix(N, double_real, A, i_l, i_u, j_l, j_u);
}
