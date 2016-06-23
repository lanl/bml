#include "bml_submatrix.h"
#include "bml_introspection.h"
#include "bml_logger.h"
//#include "dense/bml_submatrix_dense.h"
#include "ellpack/bml_submatrix_ellpack.h"

#include <stdlib.h>

/** Determine element indices for submatrix, given a set of nodes/orbitals.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Hamiltonian matrix A
 * \param B Graph matrix B
 * \param nodelist List of node/orbital indeces
 * \param nsize Size of nodelist
 * \param core_halo_index List of core+halo indeces
 * \param core_pos List of core indeces in core_halo_index
 * \param vsize Size of core_halo_index and core_pos
 * \param double_jump_flag Flag to use double jump (0=no, 1=yes)
 */
void
bml_matrix2submatrix_index(
    const bml_matrix_t * A,
    const bml_matrix_t * B,
    const int *nodelist,
    const int nsize,
    int *core_halo_index,
    int *core_pos,
    int *vsize,
    const int double_jump_flag)
{
    switch (bml_get_type(A))
    {
        case dense:
            //bml_matrix2submatrix_index_dense(A, B, nodelist, nsize,
            //    core_halo_index, core_pos, vsize, double_jump_flag);
            LOG_ERROR("bml_matrix2submatrix_index_dense NOT available\n");
            break;
        case ellpack:
            bml_matrix2submatrix_index_ellpack(A, B, nodelist, nsize,
                                               core_halo_index, core_pos,
                                               vsize, double_jump_flag);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Extract a submatrix from a matrix given a set of core+halo rows.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A
 * \param B Submatrix B
 * \param core_halo_index Set of row indeces for submatrix
 * \param llsize Number of indeces
 */
void
bml_matrix2submatrix(
    const bml_matrix_t * A,
    bml_matrix_t * B,
    const int *core_halo_index,
    const int lsize)
{
    switch (bml_get_type(A))
    {
        case dense:
            //bml_matrix2submatrix_dense(A, B, core_halo_index, lsize);
            LOG_ERROR("bml_matrix2submatrix_dense NOT available\n");
            break;
        case ellpack:
            bml_matrix2submatrix_ellpack(A, B, core_halo_index, lsize);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Assemble submatrix into a full matrixi based on core+halo indeces.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Submatrix A
 * \param B Matrix B
 * \param core_halo_index Set of submatrix row indeces
 * \param lsize Number of indeces
 * \param core_pos Set of positions in core_halo_index for core rows
 * \param llsize Number of core positions
 */
void
bml_submatrix2matrix(
    const bml_matrix_t * A,
    bml_matrix_t * B,
    const int *core_halo_index,
    const int lsize,
    const int *core_pos,
    const int llsize,
    const double threshold)
{
    switch (bml_get_type(B))
    {
        case dense:
            //bml_submatrix2matrix_dense(A, B, core_halo_index, lsize,
            //                           core_pos, llsize, threshold);
            LOG_ERROR("bml_submatrix2matrix_dense NOT available\n");
            break;
        case ellpack:
            bml_submatrix2matrix_ellpack(A, B, core_halo_index, lsize,
                                         core_pos, llsize, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** reads matrix and fills up the data structures to be used in metis and sim annealing algorithms **/

void 
bml_adjacency(
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
