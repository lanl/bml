#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_parallel_ellsort.h"
#include "bml_types_ellsort.h"

#include <stdlib.h>
#include <string.h>

/** Gather bml matrix across MPI ranks.
 *
 *  \ingroup parallel_group
 *
 *  \param A The matrix
 */
void
bml_allGatherVParallel_ellsort(
    bml_matrix_ellsort_t * A)
{

    switch (A->matrix_precision)
    {
        case single_real:
            bml_allGatherVParallel_ellsort_single_real(A);
            break;
        case double_real:
            bml_allGatherVParallel_ellsort_double_real(A);
            break;
        case single_complex:
            bml_allGatherVParallel_ellsort_single_complex(A);
            break;
        case double_complex:
            bml_allGatherVParallel_ellsort_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

#ifdef DO_MPI
void
bml_mpi_type_create_struct_ellsort(
    bml_matrix_ellsort_t * A,
    MPI_Datatype * newtype)
{

    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_type_create_struct_ellsort_single_real(A, newtype);
            break;
        case double_real:
            bml_mpi_type_create_struct_ellsort_double_real(A, newtype);
            break;
        case single_complex:
            bml_mpi_type_create_struct_ellsort_single_complex(A, newtype);
            break;
        case double_complex:
            bml_mpi_type_create_struct_ellsort_double_complex(A, newtype);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_mpi_send_ellsort(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_send_ellsort_single_real(A, dst, comm);
            break;
        case double_real:
            bml_mpi_send_ellsort_double_real(A, dst, comm);
            break;
        case single_complex:
            bml_mpi_send_ellsort_single_complex(A, dst, comm);
            break;
        case double_complex:
            bml_mpi_send_ellsort_double_complex(A, dst, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_mpi_recv_ellsort(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_recv_ellsort_single_real(A, dst, comm);
            break;
        case double_real:
            bml_mpi_recv_ellsort_double_real(A, dst, comm);
            break;
        case single_complex:
            bml_mpi_recv_ellsort_single_complex(A, dst, comm);
            break;
        case double_complex:
            bml_mpi_recv_ellsort_double_complex(A, dst, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_mpi_irecv_ellsort(
    bml_matrix_ellsort_t * A,
    const int dst,
    MPI_Comm comm)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_irecv_ellsort_single_real(A, dst, comm);
            break;
        case double_real:
            bml_mpi_irecv_ellsort_double_real(A, dst, comm);
            break;
        case single_complex:
            bml_mpi_irecv_ellsort_single_complex(A, dst, comm);
            break;
        case double_complex:
            bml_mpi_irecv_ellsort_double_complex(A, dst, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_mpi_irecv_complete_ellsort(
    bml_matrix_ellsort_t * A)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_irecv_complete_ellsort_single_real(A);
            break;
        case double_real:
            bml_mpi_irecv_complete_ellsort_double_real(A);
            break;
        case single_complex:
            bml_mpi_irecv_complete_ellsort_single_complex(A);
            break;
        case double_complex:
            bml_mpi_irecv_complete_ellsort_double_complex(A);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

bml_matrix_ellsort_t *
bml_mpi_recv_matrix_ellsort(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_mpi_recv_matrix_ellsort_single_real(N, M, src, comm);
            break;
        case double_real:
            return bml_mpi_recv_matrix_ellsort_double_real(N, M, src, comm);
            break;
        case single_complex:
            return bml_mpi_recv_matrix_ellsort_single_complex(N, M, src,
                                                              comm);
            break;
        case double_complex:
            return bml_mpi_recv_matrix_ellsort_double_complex(N, M, src,
                                                              comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_mpi_bcast_matrix_ellsort(
    bml_matrix_ellsort_t * A,
    const int root,
    MPI_Comm comm)
{
    // create MPI data type to avoid multiple messages
    MPI_Datatype mpi_data_type;
    bml_mpi_type_create_struct_ellsort(A, &mpi_data_type);

    MPI_Bcast(A, 1, mpi_data_type, root, comm);

    MPI_Type_free(&mpi_data_type);
}

#endif
