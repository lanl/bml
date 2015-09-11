#ifndef __BML_UTILITIES_H
#define __BML_UTILITIES_H

void bml_print_matrix(const int N,
                      const int i_l, const int i_u,
                      const int j_l, const int j_u,
                      bml_matrix_precision_t matrix_precision,
                      void *A);

#endif
