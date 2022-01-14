// l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom

#include <cuda_runtime_api.h>
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

template <typename T>
__global__ void kernel_sum(const T *src, const int height, const int width, float *res)
{
    const int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int thd_idx = x_index + y_index * width;

    extern __shared__ float sh[];

    if (x_index >= width || y_index >= height)
        return;
    
    load_curr_sh(src + thd_idx, sh);
    float sum = sum_sh(sh);
    if(threadIdx.x == 0)
        atomicAdd(res, sum);
}

template <typename T>
float cuMatSum(const T *d_src, const int height, const int width, const dim3 grid, const dim3 block)
{
    float val = 0;
    float *res = NULL;
    cudaMalloc(&res, sizeof(float));
    cudaMemcpy(&res, &val, sizeof(float), cudaMemcpyHostToDevice);

    kernel_sum<<<grid, block>>>(d_src, height, width, res);
    cudaDeviceSynchronize();

    cudaMemcpy(&val, res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(res);
    return val;
}

int main()
{
    int size = 1024 * 1024;
    char *d_data = NULL;
    cudaMalloc(&d_data, sizeof(char) * size);
    cudaMemset(d_data, 1, sizeof(char) * size);

    dim3 grid(16, 16, 1);
    dim3 block(16, 16, 1);
    std::cout << cuMatSum(d_data, 256, 256, grid, block) << std::endl;


    cudaFree(d_data);
    return 0;
}