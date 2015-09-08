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

#ifndef LATTE_MATRIX_H_
#define LATTE_MATRIX_H_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#if REALSIZE==4
  #undef REAL
  #define REAL float
  #undef Matrix
  #define Matrix Matrix4
#elif REALSIZE==8
  #undef REAL
  #define REAL double
  #undef Matrix
  #define Matrix Matrix8
#endif


#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "Kernels.h"
#undef num_threads
//#define NUM_THREADS 64	 
//#define NUM_THREADS 128
//#define NUM_THREADS 256
//#define NUM_THREADS 512
#define NUM_THREADS 1024 

#define MAX_GPUS 4

typedef struct {
  int M, N, DM, DN;
  REAL *Local;
  REAL *Device[MAX_GPUS];
} Matrix;

typedef struct {
  int N, DN;
  int *Local;
  int *Device[MAX_GPUS];
} Vector;

typedef struct {
  cusparseMatDescr_t descr;
  int M, N, K, HDIM, MSPARSE;
  int nnz;
  int* nnzTotalDevHostPtr;
  int* csrRowPtr;
  int* csrColInd;
  REAL* csrVal;
} SparseMatrix;

// Constants
#if REALSIZE==4
static REAL ZERO = 0.0f;
static REAL ONE = 1.0f;
static REAL MINUS1 = -1.0f;
static REAL TWO = 2.0f;
static REAL MINUS2 = -2.0f;
static REAL SMALL_NUMBER = 1.0e-14f;
#elif REALSIZE==8
static REAL ZERO = 0.0;
static REAL ONE = 1.0;
static REAL MINUS1 = -1.0;
static REAL TWO = 2.0;
static REAL MINUS2 = -2.0;
static REAL SMALL_NUMBER = 1.0e-14;
#endif

extern "C" void M_Initialize(int NGPU);
extern "C" void M_ShutDown();

void M_Init(Matrix &A, int M, int N);
void M_InitWithLocal(Matrix &A, REAL *iLocal, int iM, int iN);
void M_InitWithLocalVector(Vector &A, int *iLocal, int iN);
void M_InitVector(int *dvec, int ecount);

void M_Zero(Matrix A);

void M_Push(Matrix A);
void M_PushMgpu(Matrix A);
void M_PushAsync(Matrix A, int idevice,int istream);
void M_PushVectorAsync(int *dvec, int *hvec, int ecount, int idevice, int istream);
void M_PushVectorAsync(Vector A, int idevice, int istream);
void M_PushDeviceMgpu(Matrix A);
void M_Pull(Matrix A);
void M_PullMgpu(Matrix A);
void M_PullMgpu(Matrix A, int idevice);
void M_PullAsync(Matrix A, int idevice, int istream );
void M_CollectDistributeMgpu(Matrix A);

void M_Copy(Matrix A, Matrix B); // Copy A into B
void M_CopyMgpu(Matrix A, Matrix B); // Copy A into B on each GPU
void M_CopyAsync(Matrix A, Matrix B, int idevice, int istream);
void M_CopyDeviceAsync(Matrix A, Matrix B, int idevice, int istream);
void M_Add(Matrix A, Matrix B, Matrix C);

void M_AddColumn(REAL k, int j, Matrix A, Matrix B, Matrix C);
void M_SubtractColumn(REAL k, int j, Matrix A, Matrix B, Matrix C);

void M_Subtract(Matrix A, Matrix B, Matrix C); // C=A-B

void M_AssembleMgpu(Matrix A, Matrix A2, int sub );
void M_AssembleMgpu(Matrix A, Matrix A2, int sub, int d );

void M_Multiply(Matrix A, Matrix B, Matrix C);
void M_Multiply3(Matrix A, Matrix B, Matrix C);
void M_MultiplyTranspose(Matrix A, Matrix B, Matrix C);
void M_Multiply(REAL *scalar, Matrix A, Matrix B, REAL *scalar2, Matrix C);
void M_MultiplyAsync(REAL *scalar, Matrix A, Matrix B, REAL *scalar2, Matrix C, int idevice, int istream);
void M_MultiplyMgpu(REAL *scalar, Matrix A, Matrix B, REAL *scalar2, Matrix C);
void M_MultiplyMgpuStrm(REAL *scalar, Matrix A, Matrix B, REAL *scalar2, Matrix C, int istream);
void M_Multiply(REAL scalar, Matrix A, Matrix B); // B=scalar*A
void M_MultiplyAdd(REAL scalar, Matrix A, REAL scalar2, Matrix B, Matrix C); // C = scalar*A + scalar2*B
void M_MultiplySub(REAL scalar, Matrix A, REAL scalar2, Matrix B, Matrix C); // C = scalar*A - scalar2*B
void M_MultiplyAdd(REAL scalar, Matrix A, Matrix B, Matrix C); // C = scalar*A + B
void M_MultiplySub(REAL scalar, Matrix A, Matrix B, Matrix C); // C = scalar*A - B

// B = scalar * A + B using cublas(S/D)axpy
void M_MultiplyScalarSum(int i, REAL *scalar, Matrix A, Matrix B); 
void M_MultiplyScalarSum(REAL *scalar, Matrix A, Matrix B); 
void M_MultiplyScalarSumAsync(REAL *scalar, Matrix A, Matrix B, int idevice, int istream); 
void M_MultiplyScalarSumMgpu(REAL *scalar, Matrix A, Matrix B); 
void M_MultiplyScalarSumMgpuStrm(REAL *scalar, Matrix A, Matrix B, int istream); 
void M_MultiplyScalarSumMgpu(REAL *scalar, Matrix A, Matrix B, int d); 

// A = scalar * A using cublas(S/D)scal
void M_MultiplyScalar(int i, REAL *scalar, Matrix A);
void M_MultiplyScalar(REAL *scalar, Matrix A);
void M_MultiplyScalarMgpu(REAL *scalar, Matrix A);
void M_MultiplyScalarMgpu(REAL *scalar, REAL *A, int size);
void M_MultiplyScalarMgpu(REAL *scalar, Matrix A, int d);

void M_AddIdentity(Matrix a);

// Threshold a sparse matrix
int M_MatrixThresholdMgpu(REAL numthresh, int hdim, int *csrRowPtr, int n, int *csrColInd, REAL *csrVal, int nnz, int *csrRowPtr2, int* csrColInd2, REAL *csrVal2, int *flag);

REAL M_Trace(Matrix A);
REAL M_TraceMgpu(Matrix A, int idevice);
REAL M_TraceMgpu(Matrix A);
REAL M_TraceMgpu(int *csrRowPtr, int n, int *csrColInd, REAL *csrVal, int nnz);
REAL M_TraceX2(Matrix A);
REAL M_PartialTraceMgpuStrm(Matrix A, Vector jjp, int ncorerows, int istream);
REAL M_PartialTraceAsync(Matrix A, int *jjp, int ncorerows, int idevice, int istream);

// Dot product of A and B using cublas(S/D)dot
REAL M_DotProduct(Matrix A, Matrix B);
REAL M_DotProductOfColumn(int j, Matrix A, Matrix B);

REAL M_CGIterate(Matrix bo, Matrix p0, Matrix tmpmat, Matrix r0);

void M_Randomize(Matrix A);
void M_Print(Matrix A);
void M_DeallocateDevice(Matrix &A);
void M_DeallocateDeviceVector(Vector &A);
void M_DeallocateLocal(Matrix &A);
void M_DeallocateLocalVector(Vector &A);

void M_Dense2Sparse(Matrix &A, SparseMatrix &B);
void M_Sparse2Dense(SparseMatrix &A, Matrix &B);
void M_SparseMatrixAdd(REAL *scalar1, SparseMatrix &A, REAL *scalar2, SparseMatrix &B, SparseMatrix &C);
void M_SparseMatrixExchange(SparseMatrix &A, SparseMatrix &B);
void M_SparseMatrixMultiply(SparseMatrix &A, SparseMatrix &B, SparseMatrix &C);
void M_SparseMatrixThreshold(REAL numthresh, SparseMatrix &A, SparseMatrix &B);
REAL M_SparseMatrixTrace(SparseMatrix &A);

void M_InitSparseMatrix(SparseMatrix &A, int n, int msparse);
void M_DeallocateSparseMatrix(SparseMatrix &A);

void M_Wait();
void M_WaitStrm(int nStreams);
void M_WaitSingleStrm(int idevice, int istream);

void M_SubCopy(int size, int offset, Matrix A, Matrix A2, int d);
void M_SubCopyT(int size, int offset, Matrix A, Matrix A2, int d);
void M_TransBlk2(int size, Matrix A, Matrix A2, int d);

void *Allocate(const char Label[], void *Pointer, size_t Size);

extern "C" void sp2pure_nospin3(REAL bndfil, int  hdim, REAL *bo_pointer, REAL maxeval, REAL *h_pointer, REAL maxminusmin, int minsp2iter, int sp2convint);

extern "C" void sp2pure_seq_nospin3(int hdim, REAL *x0_pointer, REAL *vvx, int *jjp, int numiter, int ncorerows, int *pp, int idevice, int istream); 

extern "C" void sp2pure_seq_mult_nospin3(int nparts, REAL **sarray, int* hdim, int* ncore, int** jjp, REAL **vvx, int numiter, int *pp);

extern "C" void sp2pure_seq2_nospin3(int hdim, REAL *x0_pointer, REAL *vvx, int *jjp, int numiter, int ncorerows, int *pp, int idevice, int istream); 

extern "C" void sp2pure_sparse_nospin3(REAL bndfil, int hdim, int msparse, REAL *bo_pointer, REAL maxeval, REAL *h_pointer, REAL maxminusmin, REAL numthresh,int minsp2iter, int sp2convint);

void sp2pure_nospin4(REAL bndfil, int  hdim, REAL *bo_pointer, REAL maxeval, REAL *h_pointer, REAL maxminusmin, int minsp2iter, int sp2convint);

void sp2pure_sparse_nospin4(REAL bndfil, int  hdim, int msparse,REAL *bo_pointer, REAL maxeval, REAL *h_pointer, REAL maxminusmin, REAL numthresh, int minsp2iter, int sp2convint);

void sp2pure_spin3(REAL bndfil, int  hdim, REAL *rhoup_pointer, REAL *rhodown_pointer, REAL maxeval, REAL *hup_pointer, REAL *hdown_pointer, REAL maxminusmin, int minsp2iter, int sp2convint);

void sp2pure_seq_spin3(int  hdim, REAL *rhoup_pointer, REAL *rhodown_pointer, REAL *vvx, int* jjp, int numiter, int ncorerows, int *pp, int idevice, int istream);

void sp2pure_sparse_spin3(REAL bndfil, int hdim, int msparse, REAL *rhoup_pointer, REAL *rhodown_pointer, REAL maxeval, REAL *hup_pointer, REAL *hdown_pointer, REAL maxminusmin, REAL numthresh, int minsp2iter, int sp2convint);

void sp2fermi_init_nospin(REAL bndfil, int hdim, REAL *bo_pointer, REAL maxeval, REAL *h_pointer, REAL maxminusmin, REAL *chempot_pointer, int norecs, REAL *kbt_pointer, REAL *beta0_pointer, REAL breaktol);

void sp2fermi_init_spin(REAL bndfil, int hdim, REAL *rhoup_ptr, REAL *rhodown_ptr, REAL maxeval, REAL *hup, REAL *hdown, REAL maxminusmin, REAL *chempot_pointer, int norecs, REAL *kbt_pointer, REAL *beta0_pointer, REAL breaktol);

void sp2fermi_nospin(REAL bndfil, int hdim, REAL *bo_pointer, REAL maxeval, REAL *h_pointer, REAL maxminusmin, REAL *chempot_pointer, int norecs, int *signlist_pointer, REAL *beta0_pointer, REAL breaktol);

void sp2fermi_spin(REAL bndfil, int hdim, REAL *rhoup_ptr, REAL *rhodown_ptr, REAL maxeval, REAL *hup, REAL *hdown, REAL maxminusmin, REAL *chempot_pointer, int norecs, REAL *kbt_pointer, REAL *beta0_pointer, REAL breaktol);

void solve_matrix_cg(REAL *bo_ptr, int hdim, REAL cgtol2, int fermim);


//void TestMultiply();
//void TestAssemble();

#endif
