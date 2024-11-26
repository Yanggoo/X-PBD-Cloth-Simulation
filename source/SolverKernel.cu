#include "SolverKernel.cuh"

#include <device_launch_parameters.h>


using namespace ClothSolver;


__global__ void kernCalculatePredictPosition(glm::vec3* position, glm::vec3* predictPosition, glm::vec3* velocity, float deltaTime) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * blockDim.x * gridDim.x + x;

    glm::vec3 pos = position[index];
    velocity[index] += glm::vec3(0, -9.8, 0) * deltaTime;
    predictPosition[index] = pos + velocity[index] * deltaTime;
}

__global__ void updateVelocityAndWriteBack(
    glm::vec3* position,
    glm::vec3* predictPosition,
    glm::vec3* velocity,
    float deltaTime,
    float damping,
    int numParticles) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = y * blockDim.x * gridDim.x + x;
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    glm::vec3 vel = (predictPosition[idx] - position[idx]) / deltaTime;
    vel *= glm::clamp(1.0f - damping * deltaTime, 0.0f, 1.0f);
    velocity[idx] = vel;
    position[idx] = predictPosition[idx];
}

void ClothSolver::CalculatePredictPosition(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, glm::vec3* velocity, float deltaTime) {
    kernCalculatePredictPosition << <blocksPerGrid, threadsPerBlock >> > (position, predictPosition, velocity, deltaTime);

}

void ClothSolver::UpdateVelocityAndWriteBack(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, glm::vec3* velocity, float deltaTime, float damping, int numParticles) {
	updateVelocityAndWriteBack << <blocksPerGrid, threadsPerBlock >> > (position, predictPosition, velocity, deltaTime, damping, numParticles);
}


