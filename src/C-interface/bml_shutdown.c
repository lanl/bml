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
