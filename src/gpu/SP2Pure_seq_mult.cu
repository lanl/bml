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

#include <math.h>
#include <stdio.h>

#include "Matrix.h"

extern int nstreams;

extern "C" void sp2pure_seq_mult_nospin3(int nparts, REAL **sarray, 
                int* hdim, int* ncore, int** jjp, REAL **vvx, 
                         int numiter, int *pp)
{ 

  REAL trx, scalar;
  Matrix xtmp[nparts], x0[nparts];
  Vector xjjp[nparts];

  int istream;
  int idevice = 0;
  
  // Data transfer

  istream = 0;
  for (int p = 0; p < nparts; p++)
  {

    // Create x0, xtmp,  and xjjp  
    M_InitWithLocal(x0[p], sarray[p], hdim[p], hdim[p]);
    M_Init(xtmp[p], hdim[p], hdim[p]);
    M_InitWithLocalVector(xjjp[p], jjp[p], ncore[p]);

    // Copy x0 to GPU  
    istream = 0;
    M_PushAsync( x0[p], idevice, istream);
    M_PushVectorAsync(xjjp[p], idevice, istream);

  //}

  // SP2 algorithm

  istream = 1;
/*
  for (int p = 0; p < nparts; p++)
  {
*/

    // SP2 loop using sequence
    for (int i = 0; i < numiter; i++) {
    
      // xtmp = x0 - x0 * x0

      M_CopyAsync(x0[p], xtmp[p], idevice, istream);

      M_MultiplyAsync( &MINUS1, x0[p], x0[p], &ONE, xtmp[p], idevice, istream);

      // x0 = +/- xtmp
      scalar = ONE - TWO * (REAL)pp[i];
      M_MultiplyScalarSumAsync( &scalar, xtmp[p], x0[p], idevice, istream);

      // Calculate partial trace trace[(X-X^2]^2]
      trx = M_PartialTraceMgpuStrm(xtmp[p], xjjp[p], ncore[p], istream );
      vvx[p][i] += trx;

      //istream++;
      //if (istream == nstreams) istream = 1;
    }
/*
  }

  // Transfer data

  istream = 0;  
  for (int p = 0; p < nparts; p++) {
*/
    istream = 0;
    M_PullAsync(x0[p], idevice, istream);
  
    M_DeallocateLocal(xtmp[p]);
    M_DeallocateDevice(xtmp[p]);
    M_DeallocateDeviceVector(xjjp[p]);
    M_DeallocateDevice(x0[p]);

  }

}

