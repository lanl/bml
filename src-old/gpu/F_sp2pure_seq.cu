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
extern "C" void sp2purify_seq_(int  *hdim, int *spinon, void *bo_pointer,
             void *rhoup, void *rhodown, void *vvx, int* jjp, int *numiter, 
             int *ncorerows, int *pp, int *prec, int *idevice, int *istream) {
  if (*spinon) {
    if (*prec==4) {
#if REALSIZE==4
/*
      sp2pure_seq_spin3(*hdim, (float *)rhoup, (float *)rhodown,
                        (float *)vvx, (int *)jjp, *numiter, *ncorerows, (int *)pp);
*/
#endif
    }
    if (*prec==8) {
#if REALSIZE==8
/*
      sp2pure_seq_spin3(*hdim, (double *)rhoup, (double *)rhodown,
                        (double *)vvx, (int *)jjp, *numiter, *ncorerows, (int *)pp);
*/
#endif
    }
  }
  else {
    if (*prec==4) {
#if REALSIZE==4
      sp2pure_seq_nospin3(*hdim, (float *)bo_pointer, (float *)vvx,
                          (int *)jjp, *numiter, *ncorerows, (int *)pp, *idevice, *istream);
#endif
    }
    if (*prec==8) {
#if REALSIZE==8
      sp2pure_seq_nospin3(*hdim, (double *)bo_pointer, (double *)vvx,
                          (int *)jjp, *numiter, *ncorerows, (int *)pp, *idevice, *istream);
#endif
    }
  }
}
