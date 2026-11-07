/*
**  function: Matrix Multiplication ... GPU code with team-only
**            memory for A, B, and C blocks
**
**  HISTORY: Written by Tom Deakin, January 2021.
**           Ported to ROCm environment, Tim Mattson November 2026
*/
#include "mm_utils.h"

#include <assert.h>

#define Bsize 8

void mm_gpu_block(int Ndim, int Mdim, int Pdim, TYPE *A, TYPE *B, TYPE *C){

  /* Block size, must evenly divide the matrices */
  assert(Ndim % Bsize == 0);
  assert(Mdim % Bsize == 0);
  assert(Pdim % Bsize == 0);

  /* Number of blocks in each dimension */
  int Nblk = Ndim / Bsize;
  int Mblk = Mdim / Bsize;
  int Pblk = Pdim / Bsize;

#pragma omp target data map(to:A[0:Ndim*Pdim],B[0:Pdim*Mdim]) map(tofrom:C[0:Ndim*Mdim])
{
  /* Team local arrays */
  TYPE Awrk[Bsize*Bsize];
  TYPE Bwrk[Bsize*Bsize];

 #pragma omp target uses_allocators(omp_pteam_mem_alloc)
 #pragma omp teams distribute collapse(2) num_teams(Nblk*Mblk) \
               allocate(omp_pteam_mem_alloc: Awrk, Bwrk) private(Awrk, Bwrk) \
               thread_limit(Bsize*Bsize)
 for (int ib=0; ib < Nblk; ib++) /* Loop over blocks of C. One team per block */
    for (int jb=0; jb < Mblk; jb++)

      for (int kb=0; kb<Pblk; kb++)

#pragma omp parallel num_threads(Bsize*Bsize)
{
        /* Copy block of A into pteam memory */
        #pragma omp for collapse(2) nowait 
        for (int i=ib*Bsize; i<((ib+1)*Bsize); i++)
          for(int k=kb*Bsize; k<((kb+1)*Bsize); k++)
            Awrk[(i%Bsize)*Bsize + (k%Bsize)] = A[i*Pdim+k];

        /* Copy block of B into pteam memory */
        #pragma omp for collapse(2)
        for (int j=jb*Bsize; j<((jb+1)*Bsize); j++)
          for(int k=kb*Bsize; k<((kb+1)*Bsize); k++)
            Bwrk[(k%Bsize)*Bsize + (j%Bsize)] = B[k*Mdim+j];

        /* matrix multiply block */
        #pragma omp for collapse(2)
        for (int i=ib*Bsize; i<((ib+1)*Bsize); i++)
          for (int j=jb*Bsize; j<((jb+1)*Bsize); j++)
            for(int k=kb*Bsize; k<((kb+1)*Bsize); k++)
              /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
              *(C+(i*Mdim+j)) += Awrk[(i%Bsize)*Bsize + (k%Bsize)] * Bwrk[(k%Bsize)*Bsize + (j%Bsize)];
} // end parallel

} //end target data     
}
