#ifndef __LINALG_TOOLS__
#define __LINALG_TOOLS__

namespace linalgtools {

extern float M_Trace(const unsigned N,
                     const float* A);

extern cudaError_t GPUSTrace(const unsigned N
                             ,const float* A
                             ,float* B
                             ,cudaStream_t cuStrm=0);

extern cudaError_t GPUDTrace(const unsigned N
                             ,const double* A
                             ,double* B
                             ,cudaStream_t cuStrm=0);

extern cudaError_t GPUSTrace2(const unsigned N
                             ,const float* A
                             ,float* B
                             ,cudaStream_t cuStrm=0);


extern cudaError_t computeSnp1(const unsigned N
                               ,const float* Sig
                               ,const float* A
                               ,const float* B
                               ,float* C // Assumed to be on the device
                               ,cudaStream_t cuStrm=0);

extern cudaError_t computeSigma(unsigned Nocc
                               ,const float* TrXn
                               ,const float* TrX2n
                               ,float* Sig
                               ,cudaStream_t cuStrm=0);
};

#endif
