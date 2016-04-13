#include "bml_parallel.h"
#include "bml_logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>

static int myRank = 0;
static int nRanks = 1;
#ifdef DO_MPI
static MPI_Request* requestList;
static MPI_Fint fComm;
#endif
static int* rUsed;
static int reqCount = 0;

#ifdef DO_MPI
#ifdef SINGLE
#define REAL_MPI_TYPE MPI_FLOAT
#else
#define REAL_MPI_TYPE MPI_DOUBLE
#endif

#endif

/** Initialize.
 *
 * \ingroup init_group_C
 *
 * \param argc Number of args
 * \param argv Args
 */

/** Get number of MPI ranks.
 */
int bml_getNRanks()
{
   return nRanks;
}

/** Get local MPI rank.
 */
int bml_getMyRank()
{
   return myRank;
}

#ifdef DO_MPI
// Return Fortran Comm
MPI_Fint bml_getComm()
{
   return fComm;
}
#endif

//
/// \details
/// For now this is just a check for rank 0 but in principle it could be
/// more complex.  It is also possible to suppress practically all
/// output by causing this function to return 0 for all ranks.
int bml_printRank()
{
   if (myRank == 0) return 1;
   return 0;
}

void bml_initParallel(int* argc, char*** argv)
{
#ifdef DO_MPI
   MPI_Init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

   fComm = MPI_Comm_c2f(MPI_COMM_WORLD);

   requestList = (MPI_Request*) malloc(nRanks*sizeof(MPI_Request));
   rUsed = (int*) malloc(nRanks*sizeof(int));
   for (int i = 0; i < nRanks; i++) { rUsed[i] = 0; }
#endif
}

void bml_shutdownParallel()
{
#ifdef DO_MPI
   free(requestList);

   MPI_Finalize();
#endif
}

void bml_barrierParallel()
{
#ifdef DO_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif
}
