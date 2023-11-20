/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512


__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION
  __shared__ float bl[2*BLOCK_SIZE];
    unsigned int id = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2;
    
    if ( id+i >= size)
        bl[i] = 0.0;
    else
        bl[id] = in[id+i];

    if (i + id + blockDim.x >= size)
        bl[id + blockDim.x] = 0.0;
    else
        bl[id+ blockDim.x] = in[id + i+ blockDim.x];
    

    for (unsigned int j=1; j<= blockDim.x; j= j*2) 
    {
       __syncthreads();
       
       if (id % j == 0) 
           bl[id*2] += bl[id*2 + j];
    }

    if (id == 0) 
          out[blockIdx.x] = bl[0];

}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION
 __shared__ float bl[BLOCK_SIZE*2];
    unsigned int id = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2;

    if (id + i >= size)
        bl[id] = 0.0;
    else
        bl[id] = in[id+i];

    if (i + id + blockDim.x >= size)
        bl[id+ blockDim.x] = 0.0;
    else
        bl[ blockDim.x+id] = in[blockDim.x+id+i];


    for (unsigned int j=blockDim.x; j> 0 ; j=j/2)
    {
       __syncthreads();

       if (id< j)
           bl[id] = bl[id] +  bl[j+id];
    }

    if (id== 0) 
	out[blockIdx.x] = bl[0];


}
