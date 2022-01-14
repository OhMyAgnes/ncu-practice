#include "cuda_wrapper.h"
#include <iostream>

template <typename T>
__device__ inline void load_curr_sh(T *dataIn, float *sh)
{
    int shIdx = threadIdx.x;
    sh[shIdx] = static_cast<float>(*dataIn);
}

__device__ inline float sum_sh(float *sh)
{
    int shIdx = threadIdx.x;
    if(shIdx > 0)
        atomicAdd(sh, static_cast<float>(sh[shIdx]));
    __syncthreads();
    return sh[0];
}

static inline __device__ float atomicMax(float *const addr, const float val)
{
    if (*addr >= val)
        return *addr;

    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old = *addr_as_ui, assumed;
    do
    {
        assumed = old;
        if (__uint_as_float(assumed) >= val)
            break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);

    return __int_as_float(old);
}

template <typename T>
__global__ void kernel_sum(const T *src, const int height, const int width, float *res)
{
    const int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int thd_idx = x_index + y_index * width;

    extern __shared__ float sh[];

    if (x_index >= width || y_index >= height)
        return;
    

    {//global
        float val = static_cast<float>(src[thd_idx]);
        atomicAdd(res, val); 
    } 
    
    {   //sm                                            
        // load_curr_sh(src + thd_idx, sh);
        // float sum = sum_sh(sh);
        // if(threadIdx.x == 0)
        //     atomicMax(res, sum);
    }
}
template <typename T>
float cuMatSum(const T *d_src, const int height, const int width, const dim3 grid, const dim3 block)
{
    float val = 0;
    float *res = NULL;
    CudaSafeCall(cudaMalloc(&res, sizeof(float)));
    CudaSafeCall(cudaMemcpy(res, &val, sizeof(float), cudaMemcpyHostToDevice));

    kernel_sum<<<grid, block>>>(d_src, height, width, res);
    cudaDeviceSynchronize();

    CudaSafeCall(cudaMemcpy(&val, res, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(res);
    return val;
}

int main()
{
    int size = 1024 * 1024;
    char *d_data = NULL;
    CudaSafeCall(cudaMalloc(&d_data, sizeof(char) * size));
    cudaMemset(d_data, 1, sizeof(char) * size);

    {
        dim3 grid(1, 1, 1);
        dim3 block(1024, 1, 1);
        std::cout << cuMatSum(d_data, 1, 1024, grid, block) << std::endl;
    }

    // {
    //     dim3 grid(1, 1, 1);
    //     dim3 block(256, 1, 1);
    //     std::cout << cuMatSum(d_data, 1, 256, grid, block) << std::endl;
    // }

    // {
    //     dim3 grid(1, 1, 1);
    //     dim3 block(256, 1, 1);
    //     std::cout << cuMatSum(d_data, 1, 256, grid, block) << std::endl;
    // }

    // {
    //     dim3 grid(1, 1, 1);
    //     dim3 block(256, 1, 1);
    //     std::cout << cuMatSum(d_data, 1, 256, grid, block) << std::endl;
    // }

    cudaFree(d_data);
    return 0;
}