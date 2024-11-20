#pragma once

#include "Cloth.h"
#include "SolverKernel.cuh"
#include "ClothSolverBase.h"

class ClothSolverGPU : public ClothSolverBase
{

public:
    ClothSolverGPU();
    ~ClothSolverGPU();

    void ResponsibleFor(Cloth* cloth) override;
    void Simulate(float deltaTime) override;

private:
    glm::vec3* dev_position;
    glm::vec3* dev_predictPosition;
    glm::vec3* dev_velocity;

    dim3 blocksPerGrid;
    dim3 threadsPerBlock;

    const int threadDimX = 32;
    const int threadDimY = 32;

};

