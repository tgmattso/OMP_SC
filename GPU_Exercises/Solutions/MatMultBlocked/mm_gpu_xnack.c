/*
**  function: Matrix Multiplication ... for execution on a
**            a GPU supporting unified shared memory (such as
**            an AMD MI300A or Nvidia Grace Hopper).
**
**  USAGE:    On a system using AMD ROCm, you must enable 
**            unified shared memory through setting the 
**            HSA_XNACK environment variable to 1. 
**
**            I still use the function name mm_gpu() so to use
**            this version of the function, replace mm_gpu.o with
**            mm_gpu_xnack.o in the makefile object file list.
**
**  HISTORY: Written by Tim Mattson, November 2026.
*/
#include "mm_utils.h"

void mm_gpu(int Ndim, int Mdim, int Pdim, TYPE *A, TYPE *B, TYPE *C){
  int i, j, k;

#pragma omp target 
#pragma omp teams distribute parallel for collapse(2)
  for (i=0; i<Ndim; i++){
     for (j=0; j<Mdim; j++){
	for(k=0;k<Pdim;k++){
	   /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
	   *(C+(i*Mdim+j)) += *(A+(i*Pdim+k)) *  *(B+(k*Mdim+j));
	}
     }
  }
}
