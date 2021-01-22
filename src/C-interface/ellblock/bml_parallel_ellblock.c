#include "../bml_logger.h"
#include "../bml_parallel.h"
#include "../bml_types.h"
#include "bml_parallel_ellblock.h"
#include "bml_types_ellblock.h"

#include <stdlib.h>
#include <string.h>

#ifdef DO_MPI
void
bml_mpi_send_ellblock(
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_send_ellblock_single_real(A, dst, comm);
            break;
        case double_real:
            bml_mpi_send_ellblock_double_real(A, dst, comm);
            break;
        case single_complex:
            bml_mpi_send_ellblock_single_complex(A, dst, comm);
            break;
        case double_complex:
            bml_mpi_send_ellblock_double_complex(A, dst, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

void
bml_mpi_recv_ellblock(
    bml_matrix_ellblock_t * A,
    const int dst,
    MPI_Comm comm)
{
    switch (A->matrix_precision)
    {
        case single_real:
            bml_mpi_recv_ellblock_single_real(A, dst, comm);
            break;
        case double_real:
            bml_mpi_recv_ellblock_double_real(A, dst, comm);
            break;
        case single_complex:
            bml_mpi_recv_ellblock_single_complex(A, dst, comm);
            break;
        case double_complex:
            bml_mpi_recv_ellblock_double_complex(A, dst, comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

bml_matrix_ellblock_t *
bml_mpi_recv_matrix_ellblock(
    bml_matrix_precision_t matrix_precision,
    int N,
    int M,
    const int src,
    MPI_Comm comm)
{
    switch (matrix_precision)
    {
        case single_real:
            return bml_mpi_recv_matrix_ellblock_single_real(N, M, src, comm);
            break;
        case double_real:
            return bml_mpi_recv_matrix_ellblock_double_real(N, M, src, comm);
            break;
        case single_complex:
            return bml_mpi_recv_matrix_ellblock_single_complex(N, M, src,
                                                               comm);
            break;
        case double_complex:
            return bml_mpi_recv_matrix_ellblock_double_complex(N, M, src,
                                                               comm);
            break;
        default:
            LOG_ERROR("unknown precision\n");
            break;
    }
}

#endif
