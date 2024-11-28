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

    void CopyBackToCPU();
    
    size_t particleCount;
    int m_Substeps;
    int m_IterationNum;
    float m_ConstraintDistances;

    int NumWidth;
    int NumHeight;

    //std::vector<glm::vec3> m_PredPositions;
    std::vector<glm::vec3> m_Positions;
    //std::vector<glm::vec3> m_Velocities;
    std::vector<Particle*> m_Particles;

    glm::vec3* dev_position;
    glm::vec3* dev_predictPosition;
    glm::vec3* dev_velocity;
    glm::float32* dev_invMass;

    dim3 blocksPerGrid;
    dim3 threadsPerBlock;

    const int threadDimX = 32;
    const int threadDimY = 32;

};

