/** \file */

#ifndef __MACROS_H
#    define __MACROS_H

/** Row major access. */
#    define ROWMAJOR(i, j, N) (i) * (N) + (j)

/** Column major access. */
#    define COLMAJOR(i, j, N) (i) + (N) * (j)

#endif
