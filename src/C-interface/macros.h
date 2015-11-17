/** \file */

#ifndef __MACROS_H
#define __MACROS_H

/** Row major access. */
#define ROWMAJOR(i, j, M, N) (i) * (M) + (j)

/** Column major access. */
#define COLMAJOR(i, j, M, N) (i) + (N) * (j)

#endif
