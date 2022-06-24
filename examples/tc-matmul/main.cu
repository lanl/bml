extern "C"
{
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <bml_import_mptc.cu>
//#include <bml_allocate.h>
//#include <bml_utilities.h>
#include <bml.h>
}


float *bml_import_from_dense_mptc (float *);
float *bml_deallocate_mptc (cublasHandle_t, float *);
float *bml_multiply_x2_mptc (float *, const int);

// Construct H (dense type)
void
produce_hamiltonian (const unsigned N, float *X)
{
  for (int i = 0; i < N; ++i)
    {
      for (int j = i; j < N; ++j)
	{
	  X[i + j * N] =
	    exp (-0.5f * abs ((float) (i - j))) * sin ((float) (i + 1));
	  X[j + i * N] = X[i + j * N];
	}
    }
};

int
main ()
{

  float *H, *A_bml, *A2_bml;
  int N = 100;
  H = (float *) malloc (N * N * sizeof (float));
  produce_hamiltonian (N, H);

// Send to gpu TC 
  A_bml = bml_import_from_dense_mptc (H);

// Do A^2
  A2_bml = bml_multiply_x2_mptc (A_bml, N);
  
  bml_deallocate_mptc(handle, A_bml);
  bml_deallocate_mptc(handle, A2_bml);
  return 0;
}
