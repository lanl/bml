#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bml.h"
//#include <bml_import.h>
//#include <bml_allocate.h>
//#include <bml_copy.h>
//#include <bml_utilities.h>
//#include <bml_setters.h>
//#include <bml_import_mptc.h>
//#include <bml_utilities.h>


//float *bml_import_from_dense_mptc (float *);
//float *bml_deallocate_mptc (cublasHandle_t, float *);
//float *bml_multiply_x2_mptc (float *, const int);

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
// bml_matrix_t *A_bml;

//  float *H, *A2_bml;
//  int N = 100;
//  H = (float *) malloc (N * N * sizeof (float));
//  produce_hamiltonian (N, H);

// Send to gpu TC 
// //  A_bml = bml_import_from_dense (dense, double_real, dense_column_major, N, N, H, 0.0001, sequential);

// Do A^2
//  A2_bml = bml_multiply_x2 (A_bml,A_bml, 0.001);
  
 // bml_deallocate(A_bml);
 // bml_deallocate(A2_bml);
  return 0;
}
