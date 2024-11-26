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


ClothSolverGPU::ClothSolverGPU() :
    dev_position(nullptr), dev_predictPosition(nullptr), dev_velocity(nullptr), particleCount(0)
{
    m_Substeps = 5;
    m_IterationNum = 5;
}

ClothSolverGPU::~ClothSolverGPU() {
    cudaFree(dev_position);
    cudaFree(dev_predictPosition);
    cudaFree(dev_velocity);
}

void ClothSolverGPU::ResponsibleFor(Cloth* cloth) {


    int NumWidth = cloth->m_NumWidth;
    int NumHeight = cloth->m_NumHeight;
    m_Particles.reserve(NumWidth * NumHeight);
    m_Positions.resize(NumWidth * NumHeight);
    for (size_t w = 0; w < NumWidth; w++) {
        for (size_t h = 0; h < NumHeight; h++) {
            m_Particles.push_back(&cloth->m_Particles[w * NumHeight + h]);
        }
    }

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
    // todo collision

    float deltaTimeInSubstep = deltaTime / m_Substeps;
    for (int substep = 0; substep < m_Substeps; substep++) {
        ClothSolver::CalculatePredictPosition(blocksPerGrid, threadsPerBlock, dev_position, dev_predictPosition, dev_velocity, deltaTimeInSubstep);
        cudaDeviceSynchronize();
        for (int i = 0; i < m_IterationNum; i++) {
            // todo constrains
        }

        ClothSolver::UpdateVelocityAndWriteBack(blocksPerGrid, threadsPerBlock, dev_position, dev_predictPosition, dev_velocity, deltaTimeInSubstep, 0.1f, particleCount);
        cudaDeviceSynchronize();
    }

    // todo use cudaGL
    CopyBackToCPU();
}

void ClothSolverGPU::CopyBackToCPU() {
    cudaMemcpy(&m_Positions[0], dev_position, particleCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("cudaMemcpy to position failed!");

    for (int i = 0; i < particleCount; i++) {
        if (m_Particles[i]->m_InvMass == 0.0f) continue;
        m_Particles[i]->SetPosition(m_Positions[i]);
    }
}
