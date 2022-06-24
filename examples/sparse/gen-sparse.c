#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/*

Generate a sparse N x N Hamiltonian matrix given the matrix dimension, N, and threshold value, eps.
Output to terminal given in matrix market format.

Usage: 
        ./build.sh
        ./sparse N eps



Authors: Josh Finkelstein
         Christian F. A. Negre
         
         Los Alamos National Laboratory

Date:    7/22/2021

*/ 


int main(int argc, char *argv[])
{
  //
  // Input parameters
  //  
    int N = atoi(argv[1]);
    float thresh = atof(argv[2]);

  // Set memory
    int i, j;
    float *X;
    X = (float*) malloc( N * N * sizeof(float));
 
  //
  // Generate sparse Hamiltonian matrix
  //  
    for (i = 0; i<N; ++i) {
    
      for (j = i; j<N; ++j) {
           
        X[i+j*N] = exp(-0.5f*abs((float)(i-j)))*sin((float)(i+1));
  
        X[j+i*N] = X[i+j*N];

    
        // Threshold matrix entries

          if (X[i+j*N] < thresh){
      
            X[i+j*N] = 0.;
            X[j+i*N] = 0.;
               
          }
          else {
 
            fprintf(stdout, "%d %d %10.3g\n", i+1, j+1, X[i+j*N]);

          }
       
      }
    }

    return 0;
}

