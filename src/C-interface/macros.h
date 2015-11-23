/** \file */

#ifndef __MACROS_H
#define __MACROS_H

/** Row major access.
 *
 * \param i Row index.
 * \param j Column index.
 * \param M Number of rows.
 * \param N Number of columns.
 */
#define ROWMAJOR(i, j, M, N) (i) * (N) + (j)

/** Column major access.
 *
 * \param i Row index.
 * \param j Column index.
 * \param M Number of rows.
 * \param N Number of columns.
 */
#define COLMAJOR(i, j, M, N) (i) + (j) * (M)

#endif
