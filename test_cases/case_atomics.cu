#include <cuda_wrapper.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define RESTRICTION_SIZE 32

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
        atomicExch(data + (i >> 5), 6); 
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

    const int n = 2 << 24;
    int *data = new int[n];

    int i;
    for (i = 0; i < n; i++)
    {
        data[i] = i % 1024 + 1;
    }

    int *dev_data;
    CudaSafeCall(cudaMalloc((void **)&dev_data, sizeof(int) * size_t(n)));
    CudaSafeCall(cudaMemset(dev_data, 0, sizeof(int) * size_t(n)));
    CudaSafeCall(cudaMemcpy(dev_data, data, n * sizeof(int), cudaMemcpyHostToDevice));

    delete []data;

    for (int i = 0; i < 1; i++)
    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((12 * 2048) / BLOCK_SIZE); 
        AtomicOnGlobalMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());


    for (int i = 0; i < 1; i++)
    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((12 * 2048) / BLOCK_SIZE);
        WarpAtomicOnGlobalMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());

    for (int i = 0; i < 1; i++)
    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((12 * 2048) / BLOCK_SIZE);
        SameAddressAtomicOnGlobalMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());

    for (int i = 0; i < 1; i++)
    {
        dim3 blocksize(BLOCK_SIZE);
        dim3 griddize((12 * 2048) / BLOCK_SIZE);
        AtomicOnSharedMem<<<griddize, blocksize>>>(dev_data, n);
        CudaSafeCall(cudaPeekAtLastError());
    }
    CudaSafeCall(cudaDeviceSynchronize());

    CudaSafeCall(cudaFree(dev_data));
    CudaSafeCall(cudaDeviceReset());
    printf("Program finished without error.\n");
    return 0;
}