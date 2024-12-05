#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace ClothSolver {
    void CalculatePredictPosition(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, const float* invMasses, glm::vec3* velocity, float deltaTime);
    void UpdateVelocityAndWriteBack(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, glm::vec3* velocity, float deltaTime, float damping, int numParticles);

    void SolveStretchConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predictPositions, const glm::vec3* positions, const float* invMasses, const float constraintsDistances, const int numWidth, const int numHeight);

    void SolveBendingConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, const float* invMasses, const int numWidth, const int numHeight, const float constraintDistance, const float compliance, float alpha);


    void SolveCollisionSphere(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, glm::vec3* positions, const float* invMasses, glm::vec3 center, float radius);
    void SolveCollisionCube(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, glm::vec3* positions, const float* invMasses, glm::vec3 center, glm::vec3 dimensions);

    void SolveCollisionParticle(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predPosition1, glm::vec3* predPosition2, const float* invMasses, const int* neighbors, const int particleCount);
}