#include "../macros.h"
#include "../typed.h"
#include "bml_parallel.h"
#include "bml_parallel_dense.h"
#include "bml_types.h"
#include "bml_types_dense.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/** Gather a bml matrix across MPI ranks.
 *
 *  \ingroup parallel_group
 *
 *  \param A The matrix
 */
void TYPED_FUNC(
    bml_allGatherVParallel_dense) (
    bml_matrix_dense_t * A)
{
    int myRank = bml_getMyRank();

    int N = A->N;

    REAL_T *A_matrix = A->matrix;

#ifdef DO_MPI

/*
    MPI_Allgatherv(A_matrix[domain->localRowMin[myRank]], 
                   domain->localElements[myRank], REAL_MPI_TYPE,
                   A_matrix, domain->localElements, 
                   domain->recvDispl, REAL_MPI_TYPE, ccomm);  
*/

#endif

}
