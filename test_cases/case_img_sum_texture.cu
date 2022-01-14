// l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom

#include "cuda_wrapper.h"
#include <iostream>
#include <vector>

__global__ void kernel_sum(cudaTextureObject_t tex, const int height, const int width, float *res)
{
    const int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_index >= width || y_index >= height)
        return;

    atomicAdd(res, tex2D<float>(tex, x_index, y_index));
}

float cuMatSum(cudaTextureObject_t tex, const int height, const int width, const dim3 grid, const dim3 block)
{
    float val = 0;
    float *res = NULL;
    CudaSafeCall(cudaMalloc(&res, sizeof(float)));
    CudaSafeCall(cudaMemcpy(res, &val, sizeof(float), cudaMemcpyHostToDevice));

    kernel_sum<<<grid, block>>>(tex, height, width, res);
    CudaSafeCall(cudaDeviceSynchronize());

    CudaSafeCall(cudaMemcpy(&val, res, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(res);
    return val;
}

void cuInitTexture(cudaTextureObject_t &tex, const int width, const int height, cudaArray *cuArray)
{
    std::vector<float> hData(1.f, width * height);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));
    CudaSafeCall(cudaMemcpyToArray(cuArray, 0, 0, hData.data(), width * height * sizeof(float), cudaMemcpyHostToDevice));


    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    CudaSafeCall(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
}

void cuFreeTexture(cudaTextureObject_t &tex, cudaArray *cuArray)
{
    if (tex)
        CudaSafeCall(cudaDestroyTextureObject(tex));
    if (cuArray)
        cudaFreeArray(cuArray);
}

int main()
{
    const int width = 1024;
    const int height = 1024;

    cudaTextureObject_t tex;
    cudaArray *cuArray = NULL;

    cuInitTexture(tex, width, height, cuArray);

    dim3 grid(1, 1, 1);
    dim3 block(256, 1, 1);
    std::cout << cuMatSum(tex, 1, 256, grid, block) << std::endl;

    cuFreeTexture(tex, cuArray);

    return 0;
}
