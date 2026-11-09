/*
**  function: Matrix Multiplication ... three loop, ikj case
**            where ijk defines the order of the loops
**
**            Note that on an AMD MI300A CPU the ikj loop order
**            is an order of magnitude faster.
**
**  HISTORY: Written by Tim Mattson, July 2012. 
**           Modified to ikj case, Nov 2026
*/
#include "mm_utils.h"

void mm_cpu(int Ndim, int Mdim, int Pdim, TYPE *A, TYPE *B, TYPE *C){
  int i, j, k;

#pragma omp parallel for collapse(2)
  for (int i=0; i<Ndim; i++)
	for(int k=0;k<Pdim;k++)
     for (int j=0; j<Mdim; j++)
	   /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
	   *(C+(i*Mdim+j)) += *(A+(i*Pdim+k)) *  *(B+(k*Mdim+j));
}
