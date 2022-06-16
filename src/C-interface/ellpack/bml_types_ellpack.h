#ifndef __BML_TYPES_ELLPACK_H
#define __BML_TYPES_ELLPACK_H

#include "../bml_types.h"

#ifdef DO_MPI
#include <mpi.h>
#endif

/** ELLPACK matrix type. */
struct bml_matrix_ellpack_t
{
    /** The matrix type identifier. */
    bml_matrix_type_t matrix_type;
    /** The real precision. */
    bml_matrix_precision_t matrix_precision;
    /** The distribution mode. **/
    bml_distribution_mode_t distribution_mode;
    /** The number of rows. */
    int N;
    /** The number of columns per row. */
    int M;
    /** The value matrix. */
    void *value;
    /** The index matrix. */
    int *index;
    /** The vector of non-zeros per row */
    int *nnz;
    /** The domain decomposition when running in parallel. */
    bml_domain_t *domain;
    /** A copy of the domain decomposition. */
    bml_domain_t *domain2;

#if defined(BML_USE_CUSPARSE) || defined(BML_USE_ROCSPARSE)
/* need to ensure that this is sorted */
    int *csrRowPtr;
    int *csrColInd;
    void *csrVal;
#endif

#ifdef DO_MPI
    /** request field for MPI communications*/
    MPI_Request req;
#endif
};
typedef struct bml_matrix_ellpack_t bml_matrix_ellpack_t;

#endif
