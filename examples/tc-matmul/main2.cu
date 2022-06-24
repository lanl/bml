#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <regex>
#include <typeinfo>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <vector>
#include "tcore_hp_emulator.cuh"
#include "linalg_tools.cuh"

float * bml_multiply_x2_mptc (float *, const int);


__global__ void FtoD(float *X, double *Y, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < N * N) {
    Y[i] = double(X[i]);
    i += blockDim.x * gridDim.x; // add total number of threads to i
  }
}

__global__ void dev_Hamiltonian(float* X, int N){
int j = threadIdx.x + blockIdx.x * blockDim.x; //row number
int i = blockIdx.y; //column number
if (i < N && j < N){

    if (i <= j){
        //printf("(%d,%d) \n",i,j);
        X[i*N+j] = exp(-0.5f*abs((float(i-j))))*sin(float(i+1));
        X[i+N*j] = X[i*N+j];
    }
}
}
__global__ void dev_buildIdenity(float* X, int N){
int i = threadIdx.x + blockIdx.x * blockDim.x; //row number
int j = threadIdx.y + blockIdx.y * blockDim.y; //column number

if (i < N && j < N){
    if (i == j){
        //printf("(%d,%d) \n",i,j);
        X[i*N+j] = 1.0f;
    }else{
        X[i*N+j] = 0.0f;
    }
}
}

void produce_hamiltonian (const unsigned N, float *X) {
    for(int i=0; i<N; ++i) {
        for(int j=i; j<N; ++j) {
            X[i+j*N] = exp(-0.5f*abs((float)(i-j)))*sin((float)(i+1));
            X[j+i*N] = X[i+j*N];
        }
    }
};

double Frobenius (const unsigned N, double *X) {
    double sum=0.0;
    for(int i=0; i<N; ++i) {
        for (int j=0; j<N; j++){
            sum = sum + X[i*N+j]*X[i*N+j];
        }
    }
    return sqrt(sum);
};

void print_Smat (const unsigned m, const unsigned N, float* x) {
     for (int i=0; i<m;i++){
          for (int j=0; j<m; j++){
              std::cout << std::setprecision(15) << x[i*N+j] << " ";
          }
       std::cout << std::endl;
   };
};

int main(int argc, char *argv[])
{

    // Matrix size
    size_t N = atoi(argv[1]);
    size_t Nocc = atoi(argv[2]);

    int Stopp = 0;
    int iter = 0;

    std::vector<float> Idemp_Error;
     
    // Set GPU
    int device = 0;
    cudaSetDevice(device);

    // Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Set math mode
    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // Declare Memory,
    double *T, *D, *d_D, *d_TrD, *TrD, *d_T, *d_T2, *d_T4,
           *d_H, *d_energy, *energy, *comm_err, *idem_err, 
           *occ_err, *d_occ_err, *d_idem_err, *d_comm_err, 
           *d_Idd;
    float  *I, *d_S, *d_S2, *d_TrS, *d_TrS2, *d_Sig, *d_Id, 
           *sbuf1, *sbuf2, *TrS, *TrS2, *Sig, *S, *max_eigen;
    half   *hbuf1, *hbuf2;
    int    *v_sgn;
    
    // Allocate host memory
    S = (float*) malloc(N * N * sizeof(float));
    D = (double*) malloc(N * N * sizeof(double));
    I = (float*) malloc(N * N * sizeof(float));
    T = (double*) malloc( N * N * sizeof(double) );
    v_sgn = (int*) malloc( N * sizeof(int) );
    TrS = (float*) malloc(sizeof(float));
    TrS2 = (float*) malloc(sizeof(float));
    Sig = (float*) malloc(sizeof(float));
    max_eigen = (float*) malloc(sizeof(float) );
    TrD = (double*) malloc(sizeof(double) );
    energy = (double*) malloc(sizeof(double));
    comm_err = (double*) malloc(sizeof(double));
    occ_err = (double*) malloc(sizeof(double));
    idem_err = (double*) malloc(sizeof(double));
    

    // Allocate device memory
    cudaMalloc(&d_H,N*N*sizeof(double));
    cudaMalloc(&d_T,N*N*sizeof(double));
    cudaMalloc(&d_T2,N*N*sizeof(double));
    cudaMalloc(&d_T4,N*N*sizeof(double));
    cudaMalloc(&d_D,N*N*sizeof(double));
    cudaMalloc(&d_S,N*N*sizeof(float));
    cudaMalloc(&d_S2,N*N*sizeof(float));
    cudaMalloc(&d_Id,N*N*sizeof(float));
    cudaMalloc(&d_Idd,N*N*sizeof(double));
    cudaMalloc(&d_Sig,sizeof(float));
    cudaMalloc(&d_TrS,sizeof(float));
    cudaMalloc(&d_TrS2,sizeof(float));
    cudaMalloc(&d_TrD,sizeof(double));
    cudaMalloc(&d_occ_err,sizeof(double));
    cudaMalloc(&d_idem_err,sizeof(double));
    cudaMalloc(&d_energy,sizeof(double));
    cudaMalloc(&d_comm_err,sizeof(double)); 

    // Allocate Buffers
    cudaMallocManaged(&sbuf1,  N * N * sizeof(float));
    cudaMallocManaged(&sbuf2,  N * N * sizeof(float));
    cudaMallocManaged(&hbuf1,  N * N * sizeof(half));
    cudaMallocManaged(&hbuf2,  N * N * sizeof(half));
    
    // Define grid size
    int numThreads = 128;
    int numBlocks = N * N / 80 / 128 + 1; 

    // Initialize Hamiltonian and identity
    produce_hamiltonian(N, S);
    cudaMemcpy(d_S, S, N*N*sizeof(float), cudaMemcpyHostToDevice);
    //dev_Hamiltonian<<<dim3 (N / 32 + 1, N, 1), dim3 (32, 1, 1)>>>(d_S,N);
    dev_buildIdenity<<<dim3 (N, N, 1), dim3 (1, 1, 1)>>>(d_Id, N);

    // Initialize double precision matrices
    FtoD<<<numBlocks,numThreads>>>(d_S,d_H,N);
    FtoD<<<numBlocks,numThreads>>>(d_Id,d_Idd,N);
    
    // Estimate spectral bounds using power method
    //linalgtools::max_eigen(d_S, max_eigen, 1e-15, N, handle);
    float h1=-1.867;//h1 = -abs(max_eigen[0])-1;//-1.867;
    float hN=1.867; //hN = abs(max_eigen[0])+1;//1.867;
    //std::cout << "h1 = " << h1 << std::endl;
    

    // Get device id
    cudaGetDevice(&device); 
    
    //compute initial layer of the DNN, W*S+B
    float a = -1/(hN-h1); 
    float b = hN/(hN-h1); 
    cublasStat = cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &b,
                             d_Id, N,
                             d_Id, N,  
                             &a,
                             d_S, N);   //this function computes S = b*Id*Id + a*S = W*S + B
     
    // Compute initial trace
    linalgtools::GPUSTrace(N,d_S,d_TrS);
    cudaMemcpy(TrS, d_TrS, sizeof(float), cudaMemcpyDeviceToHost);  

    // SP2 DNN Loop
    std::cout << "Beginning SP2 DNN..." << std::endl;

    while (Stopp == 0) {
        
        //S^2 - half prec
       d_S2 = bml_multiply_x2_mptc (d_S, N);
	
        // Trace of S^2
        linalgtools::GPUSTrace(N,d_S2,d_TrS2); //only works for N even
        cudaMemcpy(TrS2, d_TrS2, sizeof(float), cudaMemcpyDeviceToHost); 
    
        // Idempotency error    
        Idemp_Error.push_back(TrS[0]-TrS2[0]);
        std::cout << "Idempotency error = " << Idemp_Error[iter] << std::endl;	
        
        // Convergence control
	if (TrS[0]-TrS2[0]<=0){
            break;
        };
        if (iter>2 && v_sgn[iter-1]!=v_sgn[iter-2]  && Idemp_Error[iter]>= 4.5*Idemp_Error[iter-2]*Idemp_Error[iter-2]){
            break;
        };
        
        // Compute Sigma
        linalgtools::computeSigma(Nocc,d_TrS,d_TrS2,d_Sig);

        // Compute S_{n+1}
        linalgtools::computeSnp1(N*N,d_Sig,d_S2,d_S,d_S);
       
        // Copy traces and sigma to host (seems expensive)
        cudaMemcpy(TrS2, d_TrS2, sizeof(float), cudaMemcpyDeviceToHost); 
        cudaMemcpy(Sig, d_Sig, sizeof(float), cudaMemcpyDeviceToHost); 

        // Compute TrS
        TrS[0] = Sig[0]*TrS2[0] + (1-Sig[0])*TrS[0];
        cudaMemcpy(d_TrS, TrS, sizeof(float), cudaMemcpyHostToDevice); 
        
        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
        iter += 1;
    }
    //////////////////////////////////////////////////////
    ////////////// Refinement starts here ////////////////
    //////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    FtoD<<<numBlocks, numThreads>>>(d_S, d_T, N);
    //////////////////////////////////////////////////////

    // Compute T^2 in double prec since last update was only to S, not S^2
    double alpha_dbl=1.0, beta_dbl=0.0;
    cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_T, N,
                             d_T, N,
                             &beta_dbl,
                             d_T2, N); // this function computes T2 = alpha_dbl*T*T + beta_dbl*T2 = T^2 in double precision
    cudaDeviceSynchronize();
    cudaMemcpy(d_T4, d_T2, N * N * sizeof(double), cudaMemcpyDeviceToDevice); 
    
    //////////////////////////////////////////////////////
    ////////////// compute matrix D via GPU //////////////
    ////////////////////////////////////////////////////// 
    alpha_dbl=-1.0,beta_dbl=2.0;
    cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_T2, N,
                             d_T2, N,
                             &beta_dbl,
                             d_T4, N);  // this function computes D = 2.0*T2 - 1.0*T2*T2 in double precision
    
 
    // Move D to device and host
    cudaMemcpy(d_D, d_T4, N * N * sizeof(double), cudaMemcpyDeviceToDevice); 
    cublasGetMatrix(N, N, sizeof(double),
                d_T4, N, D, N);

    //////////////////////////////////////////////////////
    ///////// Compute occupation error via GPU ///////////
    //////////////////////////////////////////////////////
    linalgtools::GPUDTrace(N,d_D,d_TrD); //compute trace on GPU
    cudaMemcpy(TrD, d_TrD, sizeof(double), cudaMemcpyDeviceToHost);
    occ_err[0] = abs(TrD[0]-Nocc);
    //////////////////////////////////////////////////////'

    //////////////////////////////////////////////////////
    ///////////// Compute energy via GPU /////////////////
    //////////////////////////////////////////////////////
    alpha_dbl = 1.0;
    beta_dbl = 0.0;
    cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_D, N,
                             d_H, N,
                             &beta_dbl,
                             d_T, N); // set T = D*H
    linalgtools::GPUDTrace(N,d_T,d_energy);
    cudaMemcpy(energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);    
    /////////////////////////////////////////////////////// 

    ///////////////////////////////////////////////////////
    ////////// Compute commutation error on GPU ///////////
    ///////////////////////////////////////////////////////
    alpha_dbl=1.0; beta_dbl=-1.0; 
    cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_H, N,
                             d_D, N,
                             &beta_dbl,
                             d_T, N); // set T = H*D - T = HD-DH   
    //linalgtools::GPUDTrace(N,d_T,d_comm_err);
    //cudaMemcpy(comm_err, d_comm_err, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(T, d_T, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    comm_err[0] = Frobenius(N,T);     // Commutation error, most sensitive
    //////////////////////////////////////////////////////

    //////////////////////////////////////////////////////
    ///////////// Compute idem err via GPU ///////////////
    //////////////////////////////////////////////////////
    alpha_dbl = 1.0;
    beta_dbl = -1.0;
    cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_T4, N,
                             d_T4, N,
                             &beta_dbl,
                             d_D, N); // D = D*D-D
    linalgtools::GPUDTrace(N,d_D,d_idem_err);
    cudaMemcpy(idem_err, d_idem_err, sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(D, d_D, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    //idem_err[0] = Frobenius(N,D);
    /////////////////////////////////////////////////////// 

    // print errors
    std::cout << "Refinement idempotency error: " << std::setprecision(15) << idem_err[0] << std::endl;
    std::cout << "Refinement occupation error: " << std::setprecision(15) << occ_err[0] << std::endl;
    std::cout << "Refinement commutation error: " << std::setprecision(15) << comm_err[0] << std::endl;
    std::cout << "Post-refinement energy: " << energy[0] << std::endl; 
    
    //Deallocate device memory
    cudaFree(d_H);
    cudaFree(d_S);
    cudaFree(d_S2);
    cudaFree(d_T);
    cudaFree(d_T2);
    cudaFree(d_T4);
    cudaFree(d_D);
    cudaFree(d_Id);
    cudaFree(d_Idd);
    cudaFree(d_TrD);
    cudaFree(d_Sig);
    cudaFree(d_TrS);
    cudaFree(d_TrS2);
    cudaFree(d_idem_err);
    cudaFree(d_energy);
    cudaFree(d_comm_err);
    cudaFree(sbuf1);
    cudaFree(sbuf2);
    cudaFree(hbuf1);
    cudaFree(hbuf2);



    //Deallocate host memory
    free(D);
    free(T);
    free(v_sgn);
    free(TrD);
    free(TrS);
    free(TrS2);
    free(Sig);
    free(energy);
    free(comm_err);
    free(occ_err);
    free(idem_err);
 
    // Destroy handle
    cublasDestroy(handle);

    return 0;
}



