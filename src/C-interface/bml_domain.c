#include "bml_types.h"
#include "bml_logger.h"
#include "bml_allocate.h"
#include "bml_parallel.h"

#include <math.h>
#include <string.h>

/** Allocate a default domain for a bml matrix.
 *
 * \ingroup allocate_group_C
 *
 *  \param N The number of rows
 *  \param M The number of columns
 *  \param distrib_mode The distribution mode
 *  \return The domain
 */
bml_domain_t *
bml_default_domain(
    int N,
    int M,
    bml_distribution_mode_t distrib_mode)
{
    int avgExtent, nleft;
    int nRanks = bml_getNRanks();

    bml_domain_t *domain = bml_allocate_memory(sizeof(bml_domain_t));

    domain->localRowMin = bml_allocate_memory(nRanks * sizeof(int));
    domain->localRowMax = bml_allocate_memory(nRanks * sizeof(int));
    domain->localRowExtent = bml_allocate_memory(nRanks * sizeof(int));
    domain->localDispl = bml_allocate_memory(nRanks * sizeof(int));
    domain->localElements = bml_allocate_memory(nRanks * sizeof(int));

    domain->totalProcs = nRanks;
    domain->totalRows = N;
    domain->totalCols = M;

    domain->globalRowMin = 0;
    domain->globalRowMax = domain->totalRows;
    domain->globalRowExtent = domain->globalRowMax - domain->globalRowMin;

    switch (distrib_mode)
    {
        case sequential:
        {
            // Default - each rank contains entire matrix, even when running distributed
            for (int i = 0; i < nRanks; i++)
            {
                domain->localRowMin[i] = domain->globalRowMin;
                domain->localRowMax[i] = domain->globalRowMax;
                domain->localRowExtent[i] =
                    domain->localRowMax[i] - domain->localRowMin[i];
                domain->localElements[i] =
                    domain->localRowExtent[i] * domain->totalCols;
                domain->localDispl[i] = 0;
            }

        }
            break;

        case distributed:
        {
            // For completely distributed
            avgExtent = N / nRanks;
            domain->maxLocalExtent = ceil((float) N / (float) nRanks);
            domain->minLocalExtent = avgExtent;

            for (int i = 0; i < nRanks; i++)
            {
                domain->localRowExtent[i] = avgExtent;
            }
            nleft = N - nRanks * avgExtent;
            if (nleft > 0)
            {
                for (int i = 0; i < nleft; i++)
                {
                    domain->localRowExtent[i]++;
                }
            }

            /** For first rank */
            domain->localRowMin[0] = domain->globalRowMin;
            domain->localRowMax[0] = domain->localRowExtent[0];

            /** For middle ranks */
            for (int i = 1; i < (nRanks - 1); i++)
            {
                domain->localRowMin[i] = domain->localRowMax[i - 1];
                domain->localRowMax[i] =
                    domain->localRowMin[i] + domain->localRowExtent[i];
            }

            /** For last rank */
            if (nRanks > 1)
            {
                int last = nRanks - 1;
                domain->localRowMin[last] = domain->localRowMax[last - 1];
                domain->localRowMax[last] =
                    domain->localRowMin[last] + domain->localRowExtent[last];
            }

            /** Number of elements and displacement per rank */
            for (int i = 0; i < nRanks; i++)
            {
                domain->localElements[i] =
                    domain->localRowExtent[i] * domain->totalCols;
                domain->localDispl[i] =
                    (i ==
                     0) ? 0 : domain->localDispl[i - 1] +
                    domain->localElements[i - 1];
            }
        }
            break;


        default:
            LOG_ERROR("unknown distribution method\n");
            break;
    }

    return domain;
}

/** Deallocate a domain.
 *
 * \ingroup allocate_group_C
 *
 * \param D[in,out] The domain.
 */
void
bml_deallocate_domain(
    bml_domain_t * D)
{
    bml_free_memory(D->localRowMin);
    bml_free_memory(D->localRowMax);
    bml_free_memory(D->localRowExtent);
    bml_free_memory(D->localDispl);
    bml_free_memory(D->localElements);
    bml_free_memory(D);
}

/** Copy a domain.
 *
 * \param A Domain to copy
 * \param B Copy of Domain A
 */
void
bml_copy_domain(
    bml_domain_t * A,
    bml_domain_t * B)
{
    int nRanks = bml_getNRanks();

    memcpy(B->localRowMin, A->localRowMin, nRanks * sizeof(int));
    memcpy(B->localRowMax, A->localRowMax, nRanks * sizeof(int));
    memcpy(B->localRowExtent, A->localRowExtent, nRanks * sizeof(int));
    memcpy(B->localDispl, A->localDispl, nRanks * sizeof(int));
    memcpy(B->localElements, A->localElements, nRanks * sizeof(int));
}

void
bml_update_domain(
    bml_domain_t * A_domain,
    int *localPartMin,
    int *localPartMax,
    int *nnodesInPart)
{
    int myRank = bml_getMyRank();
    int nprocs = bml_getNRanks();

    for (int i = 0; i < nprocs; i++)
    {
        int rtotal = 0;
        for (int j = localPartMin[i] - 1; j <= localPartMax[i] - 1; j++)
        {
            rtotal += nnodesInPart[j];
/*
         if (bml_printRank() == 1)
           printf("rank %d localPart %d %d part %d nnodesPerPart %d rtotal %d\n",
             i, localPartMin[i], localPartMax[i], j, nnodesInPart[j-1], rtotal);
*/
        }

        if (i == 0)
            A_domain->localRowMin[0] = A_domain->globalRowMin;
        else
            A_domain->localRowMin[i] = A_domain->localRowMax[i - 1];

        A_domain->localRowMax[i] = A_domain->localRowMin[i] + rtotal;
        A_domain->localRowExtent[i] =
            A_domain->localRowMax[i] - A_domain->localRowMin[i];
        A_domain->localElements[i] =
            A_domain->localRowExtent[i] * A_domain->totalCols;

        if (i == 0)
            A_domain->localDispl[0] = 0;
        else
            A_domain->localDispl[i] =
                A_domain->localDispl[i - 1] + A_domain->localElements[i - 1];
    }

    A_domain->minLocalExtent = A_domain->localRowExtent[0];
    A_domain->maxLocalExtent = A_domain->localRowExtent[0];
    for (int i = 1; i < nprocs; i++)
    {
        if (A_domain->localRowExtent[i] < A_domain->minLocalExtent)
            A_domain->minLocalExtent = A_domain->localRowExtent[i];
        if (A_domain->localRowExtent[i] > A_domain->maxLocalExtent)
            A_domain->maxLocalExtent = A_domain->localRowExtent[i];
    }
}
