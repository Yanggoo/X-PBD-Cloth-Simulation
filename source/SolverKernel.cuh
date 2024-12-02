#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace ClothSolver {
    void CalculatePredictPosition(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, const float* invMasses, glm::vec3* velocity, float deltaTime);
    void UpdateVelocityAndWriteBack(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, glm::vec3* velocity, float deltaTime, float damping, int numParticles);

    void SolveStretchConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predictPositions, const glm::vec3* positions, const float* invMasses, const float constraintsDistances, const int numWidth, const int numHeight, int numConstraints, float epsilon, int updateDirection);

    void SolveBendingConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, const float* invMasses, const int numWidth, const int numHeight, const float constraintDistance, const float compliance, float alpha, float epsilon);

    void SolveCollisionSphere(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, const float* invMasses, glm::vec3 center, float radius);
    void SolveCollisionCube(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, const float* invMasses, glm::vec3 center, glm::vec3 dimensions);
}