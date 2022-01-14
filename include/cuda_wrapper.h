#ifndef _CUDA_WRAPPER_H
#define _CUDA_WRAPPER_H

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#ifndef CUDA_ERROR_CHECK
#define CUDA_ERROR_CHECK

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }

    return;
}

inline void __cudaCheckError(const char *file, const int line)
{
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s",
                file, line, cudaGetErrorString(err));
        //LOG_ERROR("cudaCheckError() with sync failed at %s:%i : %s",
        //	file, line, cudaGetErrorString(err));
        exit(-1);
    }

    return;
}

#endif
#endif