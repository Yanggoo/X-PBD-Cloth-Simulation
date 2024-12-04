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
    void OnInitFinish() override;
    Particle* GetParticleAtScreenPos(int mouseX, int mouseY) override;
    void setSelectedParticlePosition(Particle* SelectedParticle) override;
    void OnInputSelectParticle(Particle* SelectedParticle) override;
    void OnInputClearParticle(Particle* SelectedParticle) override;

private:
    struct ClothData {
        dim3 blocksPerGrid;
        dim3 threadsPerBlock;
        int m_width;
        int m_height;
        int m_offset;
    };

    void CopyBackToCPU();
    
    size_t particleCount;
    int m_Substeps;
    int m_IterationNum;
    float m_ConstraintDistances;

    int m_currentOffset;

    //int NumWidth;
    //int NumHeight;

    //std::vector<glm::vec3> m_PredPositions;
    std::vector<glm::vec3> host_position;
    std::vector<glm::vec3> host_predictPosition;
    std::vector<glm::vec3> host_velocity;
    std::vector<float> host_invMass;
    std::vector<float> host_lambdas;
    //std::vector<glm::vec3> m_Velocities;
    std::vector<Particle*> m_Particles;

    std::vector<ClothData> m_clothData;

    glm::vec3* dev_position;
    glm::vec3* dev_predictPosition;
    glm::vec3* dev_velocity;
    glm::float32* dev_invMass;
    glm::float32* dev_lambdas;

    //dim3 blocksPerGrid;
    //dim3 threadsPerBlock;

    const int threadDimX = 32;
    const int threadDimY = 32;
};

