/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "hardshrinkPlugin.h"
#include <cuda_fp16.h>


template <typename T_DATA>
__global__ void kernelHardshrink(
    float lambd,
    T_DATA* inputs
    T_DATA* outputs
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    T_DATA input = inputs[index];
    T_DATA lambdConverted = T_DATA(lambd);

    outputs[index] = ((input > lambd) || (input < -lambd)) ? input : T_DATA(0.0);
    __syncthreads();
}

template <typename T>
int inferenceHardshrink(
    int size,
    float lambd,
    T* inputs,
    T* outputs,
    cudaStream_t stream)
{
    const int nThreads = 512;
    int nBlocks = (int)((float)size / nThreads) + 1;

    kernelHardshrink<<<nBlocks, nThreads, 0, stream>>>(lambd, inputs, outputs);

    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                __FILE__, __LINE__, cudaGetErrorString( err ) );
        return 1;
    }
    return 0;
}

int HardshrinkPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    switch(iType){
        case DataType::kFLOAT:
            return inferenceAC(mBatchDim * batchSize, mLambd, (float*)inputs[0], (float*)outputs[0], stream);
        case DataType::kHALF:
            return inferenceAC(mBatchDim * batchSize, mLambd, (__half*)inputs[0], (__half*)outputs[0], stream);
    }
    return 1;
}
