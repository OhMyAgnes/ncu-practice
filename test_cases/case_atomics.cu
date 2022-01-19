#include <cuda_wrapper.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void AtomicOnGlobalMem(int *data, int nElem)
{
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (unsigned int i = tid; i < nElem; i += blockDim.x * gridDim.x)
    {
        atomicExch(data + i, 6); 
    }
}

__global__ void WarpAtomicOnGlobalMem(int *data, int nElem)
{
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (unsigned int i = tid; i < nElem; i += blockDim.x * gridDim.x)
    {
        atomicExch(data + (i >> 3), 6); 
    }
}

__global__ void SameAddressAtomicOnGlobalMem(int *data, int nElem)
{
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (unsigned int i = tid; i < nElem; i += blockDim.x * gridDim.x)
    {
        atomicExch(data, 6); 
    }
}

__global__ void AtomicOnSharedMem(int *data, int nElem)
{
    __shared__ int smem_data[BLOCK_SIZE];
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (unsigned int i = tid; i < nElem; i += blockDim.x * gridDim.x)
    {
        atomicExch(smem_data + threadIdx.x, data[i]);
    }
}

int main(void)
{

    const int n = BLOCK_SIZE;//= 2 << 24;
    int *data = new int[n];

    for (int i = 0; i < n; i++)
    {
        data[i] = i % BLOCK_SIZE + 1;
    }

    int *dev_data;
    CudaSafeCall(cudaMalloc((void **)&dev_data, sizeof(int) * size_t(n)));
    CudaSafeCall(cudaMemset(dev_data, 0, sizeof(int) * size_t(n)));
    CudaSafeCall(cudaMemcpy(dev_data, data, n * sizeof(int), cudaMemcpyHostToDevice));

    delete []data;

    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((n + BLOCK_SIZE - 1)/ BLOCK_SIZE); 
        AtomicOnGlobalMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());

    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((n + BLOCK_SIZE - 1)/ BLOCK_SIZE); 
        WarpAtomicOnGlobalMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());

    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((n + BLOCK_SIZE - 1)/ BLOCK_SIZE); 
        SameAddressAtomicOnGlobalMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());

    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((n + BLOCK_SIZE - 1)/ BLOCK_SIZE); 
        AtomicOnSharedMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());

    CudaSafeCall(cudaFree(dev_data));
    CudaSafeCall(cudaDeviceReset());
    printf("Program finished without error.\n");
    return 0;
}