#include "bml_shutdown.h"
#include "bml_logger.h"

#include <stdlib.h>

#ifdef DO_MPI
#include <mpi.h>
#endif

/** Shutdown.
 *
 * \ingroup shutdown_group_C
 *
 */
void 
bml_shutdown()
{
#ifdef DO_MPI
    MPI_Finalize();
#endif
}
