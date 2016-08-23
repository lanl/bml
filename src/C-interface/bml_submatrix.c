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
                                               core_halo_index,
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

/** Assemble submatrix into a full matrix based on core+halo indeces.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Submatrix A
 * \param B Matrix B
 * \param core_halo_index Set of submatrix row indeces
 * \param lsize Number of indeces
 * \param llsize Number of core positions
 */
void
bml_submatrix2matrix(
    const bml_matrix_t * A,
    bml_matrix_t * B,
    const int *core_halo_index,
    const int lsize,
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
                                         llsize, threshold);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Assemble adjacency structures from matrix based on rows.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Submatrix A
 * \param xadj index to start of each row
 * \param adjncy adjacency vector
 * \param base_flag to return 0- or 1-based
*/
void 
bml_adjacency(
    const bml_matrix_t * A,
    int * xadj,
    int * adjncy,
    const int base_flag)
{

    switch (bml_get_type(A))
    {
	case dense:
	    LOG_ERROR("bml_adjacency routine is not implemented for dense\n");
	    break;
	case ellpack:
	    bml_adjacency_ellpack(A, xadj, adjncy, base_flag);
	    break;
	default:
   	    LOG_ERROR("unknown matrix type\n");
	    break;
	}
}

/** Assemble adjacency structures from matrix based on groups of rows.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Submatrix A
 * \param hindex Index for each node element
 * \param nnodes Number of groups
 * \param xadj index to start of each row
 * \param adjncy adjacency vector
 * \param base_flag return 0- or 1-based
 */
void
bml_adjacency_group(
    const bml_matrix_t * A,
    const int * hindex,
    const int nnodes,
    int * xadj,
    int * adjncy,
    const int base_flag)
{

    switch (bml_get_type(A))
    {
        case dense:
            LOG_ERROR("bml_adjacency_group routine is not implemented for dense\n");
            break;
        case ellpack:
            bml_adjacency_group_ellpack(A, hindex, nnodes, xadj, adjncy, base_flag);
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
        }
}
