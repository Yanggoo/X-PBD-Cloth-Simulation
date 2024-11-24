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


    int NumWidth = cloth->m_NumWidth;
    int NumHeight = cloth->m_NumHeight;

    if (cloth->m_NumWidth % threadDimX != 0) {
        fprintf(stderr, "Cloth width must be divisible by %d\n", threadDimX);
    }

    if (cloth->m_NumHeight % threadDimY != 0) {
        fprintf(stderr, "Cloth height must be divisible by %d\n", threadDimY);
    }

    threadsPerBlock = dim3(threadDimX, threadDimY, 1);
    blocksPerGrid = dim3(cloth->m_NumWidth / threadDimX, cloth->m_NumHeight / threadDimY, 1);

    particleCount = static_cast<size_t>(cloth->m_NumWidth) * cloth->m_NumHeight;
    cudaMalloc((void**)&dev_position, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_position failed!");

    cudaMalloc((void**)&dev_predictPosition, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_predictPosition failed!");


    cudaMalloc((void**)&dev_velocity, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_velocity failed!");


    glm::vec3* host_position = new glm::vec3[particleCount];
    glm::vec3* host_predictPosition = new glm::vec3[particleCount];
    glm::vec3* host_velocity = new glm::vec3[particleCount];

    for (size_t i = 0; i < NumWidth; i++) {
        for (size_t j = 0; j < NumHeight; j++) {
            size_t idx = i * NumHeight + j;
            host_position[idx] = cloth->m_Particles[idx].GetPosition();
            host_predictPosition[idx] = cloth->m_Particles[idx].GetPosition();
            host_velocity[idx] = glm::vec3(0.0f);
        }
    }

    cudaMemcpy(dev_position, host_position, particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy to dev_position failed!");

    cudaMemcpy(dev_predictPosition, host_predictPosition, particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy to dev_predictPosition failed!");

    cudaMemcpy(dev_velocity, host_velocity, particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy to dev_velocity failed!");

    delete[] host_position;
    delete[] host_predictPosition;
    delete[] host_velocity;

    // Add constraints here

    cudaDeviceSynchronize();
}

void ClothSolverGPU::Simulate(float deltaTime) {
    ClothSolver::CalculatePredictPosition(blocksPerGrid, threadsPerBlock, dev_position, dev_predictPosition, dev_velocity, deltaTime);
    
    cudaDeviceSynchronize();


    ClothSolver::UpdateVelocityAndWriteBack(blocksPerGrid, threadsPerBlock, dev_position, dev_predictPosition, dev_velocity, deltaTime, 0.1f, particleCount);
    
    cudaDeviceSynchronize();
}
