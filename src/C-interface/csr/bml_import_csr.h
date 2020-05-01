#ifndef __BML_IMPORT_CSR_H
#define __BML_IMPORT_CSR_H

#include "bml_types_csr.h"

bml_matrix_csr_t *bml_import_from_dense_csr(
    bml_matrix_precision_t matrix_precision,
    bml_dense_order_t order,
    int N,
    void *A,
    double threshold,
    int M,
    bml_distribution_mode_t distrib_mode);

bml_matrix_csr_t
    * bml_import_from_dense_csr_single_real(bml_dense_order_t order,
                                                int N,
                                                void *A,
                                                double threshold,
                                                int M,
                                                bml_distribution_mode_t
                                                distrib_mode);

bml_matrix_csr_t
    * bml_import_from_dense_csr_double_real(bml_dense_order_t order,
                                                int N,
                                                void *A,
                                                double threshold,
                                                int M,
                                                bml_distribution_mode_t
                                                distrib_mode);

bml_matrix_csr_t
    * bml_import_from_dense_csr_single_complex(bml_dense_order_t order,
                                                   int N,
                                                   void *A,
                                                   double threshold,
                                                   int M,
                                                   bml_distribution_mode_t
                                                   distrib_mode);

bml_matrix_csr_t
    * bml_import_from_dense_csr_double_complex(bml_dense_order_t order,
                                                   int N,
                                                   void *A,
                                                   double threshold,
                                                   int M,
                                                   bml_distribution_mode_t
                                                   distrib_mode);

#endif
