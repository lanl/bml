/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright 2010.  Los Alamos National Security, LLC. This material was    !
! produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos !
! National Laboratory (LANL), which is operated by Los Alamos National     !
! Security, LLC for the U.S. Department of Energy. The U.S. Government has !
! rights to use, reproduce, and distribute this software.  NEITHER THE     !
! GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY,     !
! EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS         !
! SOFTWARE.  If software is modified to produce derivative works, such     !
! modified software should be clearly marked, so as not to confuse it      !
! with the version available from LANL.                                    !
!                                                                          !
! Additionally, this program is free software; you can redistribute it     !
! and/or modify it under the terms of the GNU General Public License as    !
! published by the Free Software Foundation; version 2.0 of the License.   !
! Accordingly, this program is distributed in the hope that it will be     !
! useful, but WITHOUT ANY WARRANTY; without even the implied warranty of   !
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General !
! Public License for more details.                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include "Matrix.h"

// Here is our fortran interface
extern "C" void sp2purify_seq_mult(int *nparts,  void *sarray, void *hnum, void *ncore,
                void *jind, void *vvx, int *niter, void *pp, int *spinon, int *prec)
{

  if (*spinon) {
    if (*prec==4) {
#if REALSIZE==4
/*
*/
#endif
    }
    if (*prec==8) {
#if REALSIZE==8
/*
*/
#endif
    }
  }
  else {
    if (*prec==4) {
#if REALSIZE==4
      sp2pure_seq_mult_nospin3(*nparts, (float **)sarray, (int *)hnum, (int *)ncore,
                          (int **)jind, (float **)vvx, *niter, (int *)pp);
#endif
    }
    if (*prec==8) {
#if REALSIZE==8
      sp2pure_seq_mult_nospin3(*nparts, (double **)sarray, (int *)hnum, (int *)ncore,
                          (int **)jind, (double **)vvx, *niter, (int *)pp);
#endif
    }
  }
}
