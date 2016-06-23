#include "bml_adjacency.h"
#include "dense/bml_add_dense.h"
#include "ellpack/bml_add_ellpack.h"

#include <stdlib.h>

/** reads matrix and fills up the data structures to be used in metis and sim annealing algorithms **/

void bml_adjacency(
	bml_matrix_t * A,
	int * xadj,
	int * adjncy)
{
	switch (bml_get_type(A))
	{
		case dense:
			LOG_ERROR("bml_adjacency routine is not implemented for dense\n");
			break;
		case ellpack:
			bml_adjacency_ellpack(A, xadj, adjncy);
			break;
		default:
			LOG_ERROR("unknown matrix type\n");
			break;
	}
}
