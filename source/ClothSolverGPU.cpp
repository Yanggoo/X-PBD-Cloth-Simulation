#include "ClothSolverGPU.h"

#include <stdio.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char* msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


ClothSolverGPU::ClothSolverGPU()
{
    
}

ClothSolverGPU::~ClothSolverGPU() {
    cudaFree(dev_position);
    cudaFree(dev_predictPosition);
}

void ClothSolverGPU::ResponsibleFor(Cloth* cloth) {
    if (cloth->m_NumWidth % threadDimX != 0) {
        fprintf(stderr, "Cloth width must be divisible by %d\n", threadDimX);
    }

    if (cloth->m_NumHeight % threadDimY != 0) {
        fprintf(stderr, "Cloth height must be divisible by %d\n", threadDimY);
    }

    threadsPerBlock = dim3(threadDimX, threadDimY, 1);
    blocksPerGrid = dim3(cloth->m_NumWidth / threadDimX, cloth->m_NumHeight / threadDimY, 1);

    size_t particleCount = static_cast<size_t>(cloth->m_NumWidth) * cloth->m_NumHeight;
    cudaMalloc((void**)&dev_position, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_position failed!");

    cudaMalloc((void**)&dev_predictPosition, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_predictPosition failed!");

    cudaMalloc((void**)&dev_velocity, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_velocity failed!");

    cudaDeviceSynchronize();
}

void ClothSolverGPU::Simulate(float deltaTime) {
    ClothSolver::CalculatePredictPosition(blocksPerGrid, threadsPerBlock, dev_position, dev_predictPosition, dev_velocity, deltaTime);
}
