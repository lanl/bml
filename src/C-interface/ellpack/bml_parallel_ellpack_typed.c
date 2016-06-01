#include "../macros.h"
#include "../typed.h"
#include "bml_parallel.h"
#include "bml_parallel_ellpack.h"
#include "bml_types.h"
#include "bml_types_ellpack.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifdef DO_MPI
#include <mpi.h>
#endif

/** Gather a bml matrix across MPI ranks.
 *
 *  \ingroup parallel_group
 *
 *  \param A The matrix A
 */
void TYPED_FUNC(
    bml_allGatherVParallel_ellpack) (
    bml_matrix_ellpack_t * A)
{
    int myRank = bml_getMyRank();

    int N = A->N;
    int M = A->M;

    int *A_nnz = (int *) A->nnz;
    int *A_index = (int *) A->index;
    bml_domain_t * A_domain = A->domain;

    REAL_T *A_value = (REAL_T *) A->value;

#ifdef DO_MPI
    // Number of non-zeros per row
    MPI_Allgatherv(&A_nnz[A_domain->localRowMin[myRank]], 
                   A_domain->localRowExtent[myRank], MPI_INT, 
                   A_nnz, A_domain->localRowExtent, A_domain->localRowMin,
                   MPI_INT, ccomm);

    // Indeces
    MPI_Allgatherv(&A_index[A_domain->localRowMin[myRank] * M], 
                   A_domain->localElements[myRank], MPI_INT, 
                   A_index, A_domain->localElements, A_domain->localDispl,
                   MPI_INT, ccomm);

    // Values
    MPI_Allgatherv(&A_value[A_domain->localRowMin[myRank] * M],
                   A_domain->localElements[myRank], REAL_MPI_TYPE, 
                   A_value, A_domain->localElements, A_domain->localDispl,
                   REAL_MPI_TYPE, ccomm);
#endif

}
