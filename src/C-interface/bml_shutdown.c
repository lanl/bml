#include "bml_shutdown.h"
#include "bml_parallel.h"

#include <stdlib.h>

/** Shutdown.
 *
 * \ingroup shutdown_group_C
 *
 */
void 
bml_shutdown()
{
    bml_shutdownParallel();
}

/** Shutdown from Fortran.
 *
 * \ingroup shutdown_group_C
 *
 */
void
bml_shutdownF()
{
    // Fortran is expected to do the MPI_Finalize
    // Future: shutdown GPUs, cublas, cusparse, etc.
}
