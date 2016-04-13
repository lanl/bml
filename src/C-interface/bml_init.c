#include "bml_init.h"
#include "bml_parallel.h"

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
    int * argc,
    char *** argv)
{
    bml_initParallel(argc, argv);
}
