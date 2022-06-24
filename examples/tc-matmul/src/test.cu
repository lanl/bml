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
#include <ctime>
#include <cmath>
#include <vector>
#include <chrono>
#include "../include/tcore_hp_emulator.cuh"
#include "../include/linalg_tools.cuh"



double Frobenius (const unsigned N, double *X) {
    double sum=0.0;
    for(int i=0; i<N; ++i) {
        for (int j=0; j<N; j++){
            sum = sum + X[i*N+j]*X[i*N+j];
        }
    }
    return sqrt(sum);
};

float trace (const unsigned N,const float *A) {
    float sum (0.0f);
    for (unsigned i = 0; i < N; ++i) {
        sum += A[i+i*N];
    }
    return sum;
};

float Dtrace (const unsigned N, const double *A) {
    double sum=0.0;
    for (unsigned i = 0; i < N; ++i) {
        sum += A[i+i*N];
    }
    return sum;
};

void produce_hamiltonian (const unsigned N, float *X) {
    for(int i=0; i<N; ++i) {
        for(int j=i; j<N; ++j) {
            X[i+j*N] = exp(-0.5f*abs((float)(i-j)))*sin((float)(i+1));
            X[j+i*N] = X[i+j*N];
        }
    }
};

void CPU_float_to_double(float* S, double* T, int N) {
    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<N; ++j) {
            T[i+j*N] = double(S[i+j*N]);
        }
    }
};

void CPU_double_to_float(double* S, float *T, int N) {
    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<N; ++j) {
            T[i+j*N] = float(S[i+j*N]);
        }
    }
};
void matmul(double xi, double* X, double* Y, double zeta, double* Z, int N) {
    //double* z;
    //z = (double*) malloc(N*N*sizeof(double));
    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<N; ++j) {
        double local_sum = 0.0;    
            for (size_t k=0; k<N;k++){
            	local_sum += xi*X[i*N+k]*Y[k*N+j];
            }
        Z[i*N+j] = local_sum+zeta*Z[i*N+j];
        }
    }
    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<N; ++j) {
            //Z[i*N+j] = z[i*N+j];
        }
    }
    
};

void build_identity (const unsigned N, float *X) {

    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<N; ++j) {
            if (i==j){
                X[i*N+j] = 1;
            }else{
                X[i*N+j] = 0;
            }
        }
    }
};

void print_mat (const unsigned m, const unsigned n, double* x) {
    for (int i=n-m; i<n;i++){
         for (int j=n-m; j<n; j++){
             std::cout << std::setprecision(15) << x[j*n+i] << " ";
         }
         std::cout << std::endl;
     };
};
void print_Smat (const unsigned m, const unsigned n, float* x) {
    for (int i=n-m; i<n;i++){
         for (int j=n-m; j<n; j++){
             std::cout << std::setprecision(15) << x[i*n+j] << " ";
         }
         std::cout << std::endl;
     };
};

void print_mat2csv (const unsigned n, const double *x){
      std::ofstream myfile;
      myfile.open ("mat.csv");
      for (int i=0; i<n;i++){
         for (int j=0; j<n; j++){
             if (j<n-1){
                 myfile << x[i*n+j] << ", ";
             }else{
                 myfile << x[i*n+j];
             };
         };
         myfile << "\n";
     };
     myfile.close();}



int main(int argc, char *argv[])
{

    // Matrix size
    size_t N = atoi(argv[1]);
    size_t Nocc = atoi(argv[2]);

    int Stopp = 0;
    int Kvot = 0;
    int iter = 0;
    int Pur_Start = 0;
    float eps = 1e-16;

    // Prior estimate lower spectral bound
    float h1 = -1.867;
    float hN = 1.867;

    std::vector<float> Idemp_Error;
    
    std::cout << "Mat Size: " << N << std::endl;
    std::cout << "Occupied orbitals: " << Nocc << std::endl;

    // Set GPU
    int device = 0;
    cudaSetDevice(device);

    // Cublas Handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    

    cublasStatus_t cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);


    // Declare Memory,
    double *T, *d_T, *d_T2, *d_T4, *TrT, *TrT2, *D, *D2, *d_D, *D_temp, *d_TrD,*TrD, *TrD2, *d_H, *H, *Idd, *d_Idd, *d_energy, *energy, *comm_err, *idem_err, *occ_err, *d_comm_err;
    float *d_Hs, *S, *d_S, *S2, *Id, *sbuf1, *sbuf2, *sbuf3, *sbuf4, *TrS, *TrSOld, *TrS2, *Sig, *d_senergy;
    half *hbuf1, *hbuf2;
    int *v_sgn;

    float b = hN/(hN-h1); 
    float a = -1/(hN-h1);
 
    // Allocate Memory
    cudaMallocManaged(&S,     N * N * sizeof(float));
    cudaMallocManaged(&S2,     N * N * sizeof(float));
    v_sgn = (int*) malloc(N*sizeof(int));
    T = (double*) malloc(N*N*sizeof(double));
    cudaMalloc(&d_T,N*N*sizeof(double));
    cudaMalloc(&d_T2,N*N*sizeof(double));
    cudaMalloc(&d_T4,N*N*sizeof(double));
    D = (double*) malloc(N*N*sizeof(double));
    cudaMalloc(&d_D,N*N*sizeof(double));
    Idd = (double*) malloc(N*N*sizeof(double));
    cudaMalloc(&d_Idd,N*N*sizeof(double));
    H = (double*) malloc(N*N*sizeof(double));
    cudaMalloc(&d_H,N*N*sizeof(double));
    cudaMalloc(&d_Hs,N*N*sizeof(float));
    cudaMalloc(&d_S,N*N*sizeof(float));
    TrD = (double*) malloc(sizeof(double));
    cudaMalloc(&d_TrD,sizeof(double));
    energy = (double*) malloc(sizeof(double));
    cudaMalloc(&d_energy,sizeof(double));
    cudaMalloc(&d_senergy,sizeof(float));
    comm_err = (double*) malloc(sizeof(double));
    cudaMalloc(&d_comm_err,sizeof(double));
    

    //cudaMallocManaged(&D,     N * N * sizeof(double));
    cudaMallocManaged(&D_temp,     N * N * sizeof(double)); 
    cudaMallocManaged(&D2,     N * N * sizeof(double));    
    cudaMallocManaged(&Id,     N * N * sizeof(float));
    cudaMallocManaged(&TrS,    sizeof(float));
    cudaMallocManaged(&TrS2,    sizeof(float));
    cudaMallocManaged(&TrT,    sizeof(double));
    cudaMallocManaged(&TrT2,    sizeof(double));
    cudaMallocManaged(&TrD2,    sizeof(double));
    cudaMallocManaged(&TrSOld,    sizeof(float));
    cudaMallocManaged(&Sig,    sizeof(float));
    cudaMallocManaged(&occ_err,    sizeof(double));
    cudaMallocManaged(&idem_err,    sizeof(double));

    // Allocate Buffers
    cudaMallocManaged(&sbuf1,  N * N * sizeof(float));
    cudaMallocManaged(&sbuf2,  N * N * sizeof(float));
    cudaMallocManaged(&sbuf3,  N * N * sizeof(float));
    cudaMallocManaged(&sbuf4,  N * N * sizeof(float));
    cudaMallocManaged(&hbuf1,  N * N * sizeof(half));
    cudaMallocManaged(&hbuf2,  N * N * sizeof(half));

    // Produce Hamiltonian and Identity matrix 
    std::cout << "Loading Hamiltonian..." << std::endl;
    produce_hamiltonian(N,S);
    cudaMemcpy(d_Hs, S, N * N * sizeof(float), cudaMemcpyHostToDevice); // Send H to d_H   
    
    CPU_float_to_double(S,H,N); //change hamiltonian to double precision
    cudaMemcpy(d_H, H, N * N * sizeof(double), cudaMemcpyHostToDevice); // Send H to d_H  
    build_identity(N,Id);
    CPU_float_to_double(Id,D,N);
    
    // Get device id
    cudaGetDevice(&device); 

    
    //compute initial layer of the DNN, W*S+B
    
    cublasStat = cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &b,
                             Id, N,
                             Id, N,  
                             &a,
                             S, N);   //this function computes S = b*Id*Id + a*S = W*S + B
     
    //std::cout << S[0]<<" "<< S[1]<< " " << S[2]<<std::endl;

    // Compute initial trace
    linalgtools::GPUSTrace2(N,S,TrS);
    
    // begin timing
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    

    // SP2 DNN Loop
    std::cout << "Beginning SP2 DNN..." << std::endl;
    while (Stopp == 0) {
        cudaMemPrefetchAsync(TrS,   sizeof(float), device, NULL);
        cudaDeviceSynchronize();

        //S^2 - half prec
        tcoretools::tcoreSPGemmSymm(handle
                                   ,N
                                   ,S
                                   ,hbuf1
                                   ,hbuf2
                                   ,sbuf1
                                   ,sbuf2
                                   ,S2);

	// Trace of S^2
        linalgtools::GPUSTrace2(N,S2,TrS2);
        cudaMemPrefetchAsync(TrS2,   sizeof(float), device, NULL);
        cudaDeviceSynchronize();
        Idemp_Error.push_back(TrS[0]-TrS2[0]);
        std::cout << "Idempotency error = " << Idemp_Error[iter] << std::endl;	
        
        // Convergence Control
	if (TrS[0]-TrS2[0]<=0){
            break;
        };
        if (iter>2 && v_sgn[iter-1]!=v_sgn[iter-2]  && Idemp_Error[iter]>= 4.5*Idemp_Error[iter-2]*Idemp_Error[iter-2]){
            break;
        };

/*      if (abs(TrS2[0]-TrS[0]) < eps) {
            std::cout <<  "Converged!" << std::endl;
            Stopp = 1;
            iter -=1;
        } else {
            Idemp_Error.push_back(abs(TrS2[0]-TrS[0]) + eps);
           // std::cout << iter+1 << ") IdErr=" << Idemp_Error[iter] << std::endl;
            if (iter > 1) {
                Kvot = Idemp_Error[iter-2]/Idemp_Error[iter];
                if (abs(TrS2[0] - Nocc) < 0.1) {
                    if ((Pur_Start == 0) && (Kvot > 4)) {
                        Pur_Start = 1;
                    } else if ((Pur_Start == 1) && (Kvot < 3)) {
                        Stopp = 1;
                    }
                }
            }
        } 
*/
        // Compute Sigma
        linalgtools::computeSigma(Nocc,TrS,TrS2,Sig);
        
        // Compute S_{n+1}
        linalgtools::computeSnp1(N*N,Sig,S2,S,S);
        cudaDeviceSynchronize();
        
        // Compute TrS
        TrS[0] = Sig[0]*TrS2[0] + (1-Sig[0])*TrS[0];
        std::cout << TrS[0] << std::endl; 
        // Update sign vector
        v_sgn[iter]=int(Sig[0]);
        
        cudaMemPrefetchAsync(TrS, sizeof(float), device, NULL);
        iter += 1;
        
    }
    // Compute timing of loop
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Time difference = " << duration << "[µs]" << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
    double time = 2*iter*N*N*N/(duration/1e6);
    std::cout << time << std::endl;
    

    //////////////////////////////////////////////////////
    ////////////// Refinement starts here ////////////////
    //////////////////////////////////////////////////////
    cudaDeviceSynchronize();
    CPU_float_to_double(S,T,N);
    CPU_float_to_double(Id,Idd,N); 

    //////////////////////////////////////////////////////
    //// Send double precision object back to the GPU ///
    //////////////////////////////////////////////////////
    cudaMemcpy(d_T, T, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Idd, Idd, N * N * sizeof(double), cudaMemcpyHostToDevice); 
    //////////////////////////////////////////////////////
    
    //record time to file
    std::ofstream myfile;
    myfile.open ("timings_A100.csv", std::ios::app);
    myfile << N << ", " << time << "\n";
    myfile.close();


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
    cudaMemcpy(d_D, d_T4, N * N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(D, d_T4, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    

    //////////////////////////////////////////////////////
    ///////// Compute occupation error via GPU ///////////
    //////////////////////////////////////////////////////
    linalgtools::GPUDTrace(N,d_D,d_TrD); //compute trace on GPU
    cudaMemcpy(TrD, d_TrD, sizeof(double), cudaMemcpyDeviceToHost);
    occ_err[0] = abs(TrD[0]-Nocc);
    //std::cout << "occ error tr(D) = " << std::setprecision(15) << TrD[0] << std::endl; 
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
    cudaMemcpy(T, d_T, N*N*sizeof(double), cudaMemcpyDeviceToHost);    
    linalgtools::GPUDTrace(N,d_T,d_energy);
    cudaMemcpy(energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);    
    /////////////////////////////////////////////////////// 


    ///////////////////////////////////////////////////////
    ////////// Compute commutation error on GPU ///////////
    ///////////////////////////////////////////////////////
    comm_err[0]=1.0;
    alpha_dbl=-1.0; beta_dbl=1.0; 
    cublasStat = cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_dbl,
                             d_H, N,
                             d_D, N,
                             &beta_dbl,
                             d_T, N); // set T = H*D - T = HD-DH   

    cudaMemcpy(T, d_T, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    comm_err[0] = Frobenius(N,T);     // Commutation error, most sensitive
    ///////////////////////////////////////////////////////
    

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
    cudaMemcpy(D, d_D, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    idem_err[0] = Frobenius(N,D);
    /////////////////////////////////////////////////////// 

    // print errors
    std::cout << "Refinement idempotency error: " << std::setprecision(15) << idem_err[0] << std::endl;
    std::cout << "Refinement occupation error: " << std::setprecision(15) << occ_err[0] << std::endl;
    std::cout << "Refinement commutation error: " << std::setprecision(15) << comm_err[0] << std::endl;
    std::cout << "Post-refinement energy: " << energy[0] << std::endl; 
    return 0;
}



