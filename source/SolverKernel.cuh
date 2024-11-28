#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace ClothSolver {
    void CalculatePredictPosition(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, const float* invMasses, glm::vec3* velocity, float deltaTime);
    void UpdateVelocityAndWriteBack(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, glm::vec3* velocity, float deltaTime, float damping, int numParticles);

    void SolveStretchConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predictPositions, const glm::vec3* positions, const float* invMasses, const float constraintsDistances, const int numWidth, const int numHeight, int numConstraints, float epsilon, int updateDirection);



}