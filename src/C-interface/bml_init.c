#include "bml_init.h"
#include "bml_parallel.h"

#ifdef BML_USE_MAGMA
#include "magma_v2.h"
#endif

#include <stdlib.h>

/** Initialize.
 *
 * \ingroup init_group_C
 *
 * \param argc Number of args
 * \param argv Args
 */
void
bml_init(
    int *argc,
    char ***argv)
{
#ifdef BML_USE_MAGMA
    magma_init();
#endif
    bml_initParallel(argc, argv);
}

/** Initialize from Fortran.
 *
 * \ingroup init_group_C
 *
 * \param Comm from Fortran
 */
void
bml_initF(
    int fcomm)
{
    bml_initParallelF(fcomm);
}
