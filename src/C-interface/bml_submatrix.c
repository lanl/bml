#include "bml_submatrix.h"
#include "bml_introspection.h"
#include "bml_logger.h"
#include "csr/bml_submatrix_csr.h"
#include "dense/bml_submatrix_dense.h"
#include "ellpack/bml_submatrix_ellpack.h"
#include "ellblock/bml_submatrix_ellblock.h"

#include <stdlib.h>

/** Determine element indices for submatrix, given a set of nodes/orbitals.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Hamiltonian matrix A
 * \param B Graph matrix B
 * \param nodelist List of node/orbital indices
 * \param nsize Size of nodelist
 * \param core_halo_index List of core+halo indices
 * \param vsize Size of core_halo_index and core_pos
 * \param double_jump_flag Flag to use double jump (0=no, 1=yes)
 */
void
bml_matrix2submatrix_index(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int *nodelist,
    int nsize,
    int *core_halo_index,
    int *vsize,
    int double_jump_flag)
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
        case ellblock:
            LOG_ERROR("bml_matrix2submatrix_index_ellblock NOT available\n");
            break;
        case csr:
            LOG_ERROR("bml_matrix2submatrix_index_csr NOT available\n");
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Determine element indices for submatrix, given a set of nodes/orbitals.
 *
 * \ingroup submatrix_group_C
 *
 * \param B Graph matrix B
 * \param nodelist List of node/orbital indices
 * \param nsize Size of nodelist
 * \param core_halo_index List of core+halo indices
 * \param vsize Size of core_halo_index and core_pos
 * \param double_jump_flag Flag to use double jump (0=no, 1=yes)
 */
void
bml_matrix2submatrix_index_graph(
    bml_matrix_t * B,
    int *nodelist,
    int nsize,
    int *core_halo_index,
    int *vsize,
    int double_jump_flag)
{
    switch (bml_get_type(B))
    {
        case dense:
            //bml_matrix2submatrix_index_graph_dense(B, nodelist, nsize,
            //    core_halo_index, core_pos, vsize, double_jump_flag);
            LOG_ERROR("bml_matrix2submatrix_index_dense NOT available\n");
            break;
        case ellpack:
            bml_matrix2submatrix_index_graph_ellpack(B, nodelist,
                                                     nsize, core_halo_index,
                                                     vsize, double_jump_flag);
            break;
        case ellblock:
            LOG_ERROR("bml_matrix2submatrix_index_ellblock NOT available\n");
            break;
        case csr:
            LOG_ERROR("bml_matrix2submatrix_index_csr NOT available\n");
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Assemble matrix based on groups of rows from a matrix.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A
 * \param hindex Indices of nodes
 * \param ngroups Number of groups
 * \param threshold Threshold for graph
 */
bml_matrix_t *
bml_group_matrix(
    bml_matrix_t * A,
    int *hindex,
    int ngroups,
    double threshold)
{
    switch (bml_get_type(A))
    {
        case dense:
            //return bml_group_matrix_dense(A, hindex, ngroups, threshold);
            LOG_ERROR("bml_group_matrix_dense NOT available\n");
            break;
        case ellpack:
            return bml_group_matrix_ellpack(A, hindex, ngroups, threshold);
            break;
        case ellblock:
            LOG_ERROR("bml_group_matrix_ellblock NOT available\n");
            break;
        case csr:
            LOG_ERROR("bml_group_matrix_csr NOT available\n");
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
    return NULL;
}

/** Extract a submatrix from a matrix given a set of core+halo rows.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Matrix A
 * \param B Submatrix B
 * \param core_halo_index Set of row indices for submatrix
 * \param llsize Number of indices
 */
void
bml_matrix2submatrix(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int *core_halo_index,
    int lsize)
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
        case ellblock:
            LOG_ERROR("bml_matrix2submatrix_ellblock NOT available\n");
            break;
        case csr:
            LOG_ERROR("bml_matrix2submatrix_csr NOT available\n");
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

/** Assemble submatrix into a full matrix based on core+halo indices.
 *
 * \ingroup submatrix_group_C
 *
 * \param A Submatrix A
 * \param B Matrix B
 * \param core_halo_index Set of submatrix row indices
 * \param lsize Number of indices
 * \param llsize Number of core positions
 */
void
bml_submatrix2matrix(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int *core_halo_index,
    int lsize,
    int llsize,
    double threshold)
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
        case ellblock:
            LOG_ERROR("bml_submatrix2matrix_ellblock NOT available\n");
            break;
        case csr:
            LOG_ERROR("bml_submatrix2matrix_csr NOT available\n");
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
    bml_matrix_t * A,
    int *xadj,
    int *adjncy,
    int base_flag)
{

    switch (bml_get_type(A))
    {
        case dense:
            LOG_ERROR("bml_adjacency routine is not implemented for dense\n");
            break;
        case ellpack:
            bml_adjacency_ellpack(A, xadj, adjncy, base_flag);
            break;
        case ellblock:
            LOG_ERROR
                ("bml_adjacency routine is not implemented for ellblock\n");
            break;
        case csr:
            LOG_ERROR("bml_adjacency routine is not implemented for csr\n");
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
    bml_matrix_t * A,
    int *hindex,
    int nnodes,
    int *xadj,
    int *adjncy,
    int base_flag)
{

    switch (bml_get_type(A))
    {
        case dense:
            LOG_ERROR
                ("bml_adjacency_group routine is not implemented for dense\n");
            break;
        case ellpack:
            bml_adjacency_group_ellpack(A, hindex, nnodes, xadj, adjncy,
                                        base_flag);
            break;
        case ellblock:
            LOG_ERROR
                ("bml_adjacency_group routine is not implemented for ellblock\n");
            break;
        case csr:
            LOG_ERROR
                ("bml_adjacency_group routine is not implemented for csr\n");
            break;
        default:
            LOG_ERROR("unknown matrix type\n");
            break;
    }
}

bml_matrix_t *
bml_extract_submatrix(
    bml_matrix_t * A,
    int irow,
    int icol,
    int B_N,
    int B_M)
{
    bml_matrix_t *B = NULL;

    switch (bml_get_type(A))
    {
        case ellpack:
            B = bml_extract_submatrix_ellpack(A, irow, icol, B_N, B_M);
            break;
        case dense:
            B = bml_extract_submatrix_dense(A, irow, icol, B_N, B_M);
            break;
        case csr:
            B = bml_extract_submatrix_csr(A, irow, icol, B_N, B_M);
            break;
        case ellblock:
            B = bml_extract_submatrix_ellblock(A, irow, icol, B_N, B_M);
            break;
        default:
            LOG_ERROR
                ("bml_extract_submatrix not implemented for matrix format\n");
            break;
    }
    return B;
}

void
bml_assign_submatrix(
    bml_matrix_t * A,
    bml_matrix_t * B,
    int irow,
    int icol)
{
    switch (bml_get_type(A))
    {
        case ellpack:
            bml_assign_submatrix_ellpack(A, B, irow, icol);
            break;
        case dense:
            bml_assign_submatrix_dense(A, B, irow, icol);
            break;
        case csr:
            bml_assign_submatrix_csr(A, B, irow, icol);
            break;
        case ellblock:
            bml_assign_submatrix_ellblock(A, B, irow, icol);
            break;
        default:
            LOG_ERROR
                ("bml_assign_submatrix not implemented for matrix format\n");
            break;
    }
}
