#include "bml.h"
#include "../typed.h"
#include "bml_introspection.h"

#include <stdlib.h>
#include <stdio.h>

int TYPED_FUNC(
    test_introspection) (
    const int N,
    const bml_matrix_type_t matrix_type,
    const bml_matrix_precision_t matrix_precision,
    const int M)
{
    bml_matrix_t *A = NULL;

    bml_distribution_mode_t distrib_mode = sequential;
#ifdef DO_MPI
    if (bml_getNRanks() > 1)
    {
        LOG_INFO("Use distributed matrix\n");
        distrib_mode = distributed;
    }
#endif

    A = bml_random_matrix(matrix_type, matrix_precision, N, M, distrib_mode);

    bml_matrix_type_t deep_type = bml_get_deep_type(A);

    bml_matrix_type_t type = bml_get_type(A);

#ifdef DO_MPI
    switch (deep_type)
    {
        case dense:
            LOG_INFO("deep type is dense\n");
            break;
        case ellpack:
            LOG_INFO("deep type is ellpack\n");
            break;
        case ellsort:
            LOG_INFO("deep type is ellsort\n");
            break;
        case ellblock:
            LOG_INFO("deep type is ellblock\n");
            break;
        case csr:
            LOG_INFO("deep type is csr\n");
            break;
        case distributed2d:
            LOG_ERROR("deep type should not be distributed2d\n");
            return -1;
            break;
        default:
            LOG_ERROR("unknown deep type\n");
            return -1;
            break;
    }

#else
    if (deep_type != type)
    {
        LOG_ERROR("type and deep_type not equal!\n");
        return -1;
    }
#endif

    LOG_INFO("bml_introspection passed for task %d\n", bml_getMyRank());

    bml_deallocate(&A);

    return 0;
}
