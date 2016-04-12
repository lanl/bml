#include "bml_init.h"
#include "bml_logger.h"

#include <stdlib.h>

#ifdef DO_MPI
#include <mpi.h>
#endif

/** Initialize.
 *
 * \ingroup init_group_C
 *
 * \param argc Number of args
 * \param argv Args
 */
void 
bml_init(
    int * argc,
    char *** argv)
{
#ifdef DO_MPI
    MPI_Init(argc, argv);
#endif
}
