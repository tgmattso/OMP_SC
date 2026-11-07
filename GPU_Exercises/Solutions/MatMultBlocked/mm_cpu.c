/*
**  function: Matrix Multiplication ... three loop, ikj case
**            where ijk defines the order of the loops
**
**  HISTORY: Written by Tim Mattson, July 2012. 
*/
#include "mm_utils.h"

void mm_cpu(int Ndim, int Mdim, int Pdim, TYPE *A, TYPE *B, TYPE *C){
  int i, j, k;

#pragma omp parallel for collapse(2)
  for (int i=0; i<Ndim; i++)
     for (int j=0; j<Mdim; j++)
	for(int k=0;k<Pdim;k++)
	   /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
	   *(C+(i*Mdim+j)) += *(A+(i*Pdim+k)) *  *(B+(k*Mdim+j));
}
