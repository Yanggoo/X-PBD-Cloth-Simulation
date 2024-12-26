#include "ClothSolverGPU.h"
//#include <iostream>
#include <stdio.h>
#include "Sphere.h"
#include "Cube.h"

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
    dev_position(nullptr), dev_predictPosition(nullptr), dev_predictPosition2(nullptr), dev_velocity(nullptr),
    dev_invMass(nullptr), dev_neighbors(nullptr), particleCount(0), m_currentOffset(0), m_ConstraintDistances(0), m_KDTree(host_position)
{
    m_Substeps = 5;
    m_IterationNum = 10;
}

ClothSolverGPU::~ClothSolverGPU() {
    cudaFree(dev_position);
    cudaFree(dev_predictPosition);
    cudaFree(dev_predictPosition2);
    cudaFree(dev_invMass);
    cudaFree(dev_velocity);
}

void ClothSolverGPU::ResponsibleFor(Cloth* cloth)
{
    if (cloth->m_NumWidth % threadDimX != 0) {
        fprintf(stderr, "Cloth width must be divisible by %d\n", threadDimX);
    }

    if (cloth->m_NumHeight % threadDimY != 0) {
        fprintf(stderr, "Cloth height must be divisible by %d\n", threadDimY);
    }

    ClothData clothData;
    clothData.m_width = cloth->m_NumWidth;
    clothData.m_height = cloth->m_NumHeight;
    clothData.threadsPerBlock = dim3(threadDimX, threadDimY, 1);
    clothData.blocksPerGrid = dim3(cloth->m_NumWidth / threadDimX, cloth->m_NumHeight / threadDimY, 1);
    clothData.m_offset = m_currentOffset;
    m_clothData.push_back(clothData);
    m_currentOffset += cloth->m_NumWidth * cloth->m_NumHeight;

    m_Particles.reserve(m_currentOffset);
    host_position.reserve(m_currentOffset);
    host_predictPosition.reserve(m_currentOffset);
    host_velocity.reserve(m_currentOffset);
    host_invMass.reserve(m_currentOffset);

    particleCount = m_currentOffset;

    for (size_t i = 0; i < cloth->m_NumWidth; i++) {
        for (size_t j = 0; j < cloth->m_NumHeight; j++) {
            size_t idx = i * cloth->m_NumHeight + j;
            m_Particles.push_back(&cloth->m_Particles[idx]);
            host_position.push_back(cloth->m_Particles[idx].GetPosition());
            host_predictPosition.push_back(cloth->m_Particles[idx].GetPosition());
            host_velocity.emplace_back(0.0f);
            host_invMass.push_back(cloth->m_Particles[idx].m_InvMass);
        }
    }

    // Add constraints here

    // TODO: Stretch Constraints should be seperated into horizontal and vertical constraints (Record vertical and horizontal distances constraints separately)

    //m_ConstraintDistances = new float[2];

    //m_ConstraintDistances[0] = glm::length(cloth->m_Particles[0].GetPosition() - cloth->m_Particles[1].GetPosition());
    //m_ConstraintDistances[1] = glm::length(cloth->m_Particles[0].GetPosition() - cloth->m_Particles[NumHeight].GetPosition());

    //std::cout<<m_ConstraintDistances[0]<<std::endl;
    //std::cout<<m_ConstraintDistances[1]<<std::endl;

    m_ConstraintDistances.push_back(glm::length(cloth->m_Particles[0].GetPosition() - cloth->m_Particles[1].GetPosition()));
    m_ConstraintDistances.push_back(glm::length(cloth->m_Particles[1].GetPosition() - cloth->m_Particles[cloth->m_NumWidth].GetPosition()));

}

void ClothSolverGPU::Simulate(float deltaTime) {
    {
        std::vector<int> allNeighbors;
        for (int i = 0; i < particleCount; i++) {
            auto neighbors = m_KDTree.queryNeighbors(host_position[i], COLLISION_NEIGHBOR_COUNT);
            for (int j = 0; j < COLLISION_NEIGHBOR_COUNT; j++) {
                allNeighbors.push_back(neighbors[j]);
            }
        }

        cudaMemcpy(dev_neighbors, &allNeighbors[0], particleCount * COLLISION_NEIGHBOR_COUNT * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAErrorWithLine("cudaMemcpy to dev_position failed!");
    }

    {
        for (const ClothData& clothData : m_clothData) {
            for (size_t i = 0; i < m_Colliders.size(); i++) {
                Collider* collider = m_Colliders[i];
                dim3 blocksPerGrid = clothData.blocksPerGrid;
                dim3 threadsPerBlock = clothData.threadsPerBlock;
                int offset = clothData.m_offset;
                if (Sphere* sphere = dynamic_cast<Sphere*>(collider)) {
                    ClothSolver::SolveCollisionSphere(blocksPerGrid, threadsPerBlock, dev_position + offset, dev_position + offset, dev_invMass + offset, sphere->m_Position, sphere->m_Radius);
                }
                else if (Cube* cube = dynamic_cast<Cube*>(collider)) {
                    ClothSolver::SolveCollisionCube(blocksPerGrid, threadsPerBlock, dev_position + offset, dev_position + offset, dev_invMass + offset, cube->m_Position, cube->m_Dimensions);
                }
            }
        }
    }

    float deltaTimeInSubstep = deltaTime / m_Substeps;
    for (int substep = 0; substep < m_Substeps; substep++) {
        dim3 clothThreadPerBlock = dim3(threadDimX, 1, 1);
        dim3 clothBlockPerGrid = dim3(particleCount / threadDimX, 1, 1);
        ClothSolver::CalculatePredictPosition(clothBlockPerGrid, clothThreadPerBlock, dev_position, dev_predictPosition, dev_invMass, dev_velocity, deltaTimeInSubstep);
        cudaDeviceSynchronize();

        for (int i = 0; i < m_IterationNum; i++) {
            for (const ClothData& clothData : m_clothData) {
                dim3 blocksPerGrid = clothData.blocksPerGrid;
                dim3 threadsPerBlock = clothData.threadsPerBlock;
                int offset = clothData.m_offset;
                int width = clothData.m_width;
                int height = clothData.m_height;
                int clothParticleCount = width * height;
                // todo constrains
                float vecs[2] = {m_ConstraintDistances[0],m_ConstraintDistances[1]};
                ClothSolver::SolveStretchConstraints(blocksPerGrid, threadsPerBlock, dev_predictPosition + offset, dev_position + offset, dev_invMass + offset, vecs, width, height);
                cudaDeviceSynchronize();

                auto bendComplience = 10;
                auto alpha = bendComplience / (deltaTime * deltaTime + 1e-6);
                dim3 smallThreadsPerBlock = dim3(threadsPerBlock.x/2, threadsPerBlock.y/2, threadsPerBlock.z);
                ClothSolver::SolveBendingConstraints(blocksPerGrid, smallThreadsPerBlock, dev_predictPosition + offset, dev_invMass + offset, width, height, glm::radians(180.f), bendComplience, alpha);

                for (size_t i = 0; i < m_Colliders.size(); i++) {
                    Collider* collider = m_Colliders[i];
                    if (Sphere* sphere = dynamic_cast<Sphere*>(collider)) {
                        ClothSolver::SolveCollisionSphere(blocksPerGrid, threadsPerBlock, dev_predictPosition + offset, 
                            dev_position + offset, dev_invMass + offset, sphere->m_Position, sphere->m_Radius);
                    }
                    else if (Cube* cube = dynamic_cast<Cube*>(collider)) {
                        ClothSolver::SolveCollisionCube(blocksPerGrid, threadsPerBlock, dev_predictPosition + offset, 
                            dev_position + offset,  dev_invMass + offset, cube->m_Position, cube->m_Dimensions);
                    }
                }

            }

            cudaMemcpy(dev_predictPosition2, dev_predictPosition, particleCount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
            ClothSolver::SolveCollisionParticle(clothBlockPerGrid, clothThreadPerBlock, dev_position, dev_predictPosition, dev_predictPosition2, dev_invMass, dev_neighbors, particleCount);
            cudaMemcpy(dev_predictPosition, dev_predictPosition2, particleCount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

        }

        ClothSolver::UpdateVelocityAndWriteBack(clothBlockPerGrid, clothThreadPerBlock, dev_position, dev_predictPosition, dev_velocity, deltaTimeInSubstep, damping, particleCount);
        
        cudaDeviceSynchronize();
    }

    // todo use cudaGL
    CopyBackToCPU();

    m_KDTree.rebuild();
}

void ClothSolverGPU::OnInitFinish() {
    cudaMalloc((void**)&dev_position, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_position failed!");
    cudaMalloc((void**)&dev_predictPosition, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_predictPosition failed!");
    cudaMalloc((void**)&dev_predictPosition2, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_predictPosition2 failed!");
    cudaMalloc((void**)&dev_velocity, particleCount * sizeof(glm::vec3));
    checkCUDAErrorWithLine("cudaMalloc dev_velocity failed!");
    cudaMalloc((void**)&dev_invMass, particleCount * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_invMass failed!");
    cudaMalloc((void**)&dev_neighbors, particleCount * COLLISION_NEIGHBOR_COUNT * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_lambdas failed!");

    cudaMemcpy(dev_position, &host_position[0], particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy to dev_position failed!");
    cudaMemcpy(dev_predictPosition, &host_predictPosition[0], particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy to dev_predictPosition failed!");
    cudaMemcpy(dev_velocity, &host_velocity[0], particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy to dev_velocity failed!");
    cudaMemcpy(dev_invMass, &host_invMass[0], particleCount * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy to dev_invMass failed!");

    host_predictPosition.clear();
    host_velocity.clear();
    host_invMass.clear();

    cudaDeviceSynchronize();

    m_KDTree.rebuild();
}

Particle* ClothSolverGPU::GetParticleAtScreenPos(int mouseX, int mouseY) {
    glm::vec3 worldPos = Mouse2World(mouseX, mouseY);
    if (worldPos == glm::vec3(10, 10, 10)) return nullptr;

    float minDistance = 1000000;
    Particle* closestParticle = nullptr;
    for (int i = 0; i < particleCount; i++) {
        float distance = glm::length(worldPos - host_position[i]);
        if (distance < minDistance) {
            minDistance = distance;
            closestParticle = m_Particles[i];
        }
    }
    return closestParticle;
}

void ClothSolverGPU::setSelectedParticlePosition(Particle* SelectedParticle) {
    if (SelectedParticle == nullptr) return;
    for (int i = 0; i < particleCount; i++) {
        if (m_Particles[i] == SelectedParticle) {
            glm::vec3 position = SelectedParticle->GetPosition();
            //m_PredPositions[i] = SelectedParticle->GetPosition();
            //m_Positions[i] = SelectedParticle->GetPosition();
            cudaMemcpy(dev_position + i, &position, sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_predictPosition + i, &position, sizeof(glm::vec3), cudaMemcpyHostToDevice);
        }
    }

    cudaDeviceSynchronize();
}

void ClothSolverGPU::OnInputSelectParticle(Particle* SelectedParticle) {
    if (SelectedParticle == nullptr) return;
    for (int i = 0; i < particleCount; i++) {
        if (m_Particles[i] == SelectedParticle) {
            float invMass = 0.0f;
            cudaMemcpy(dev_invMass + i, &invMass, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}

void ClothSolverGPU::OnInputClearParticle(Particle* SelectedParticle) {
    if (SelectedParticle == nullptr) return;
    for (int i = 0; i < particleCount; i++) {
        if (m_Particles[i] == SelectedParticle) {
            float invMass = 1.0f;
            cudaMemcpy(dev_invMass + i, &invMass, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}


void ClothSolverGPU::CopyBackToCPU() {
    cudaMemcpy(&host_position[0], dev_position, particleCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("cudaMemcpy to position failed!");

    for (int i = 0; i < particleCount; i++) {
        if (m_Particles[i]->m_InvMass == 0.0f) continue;
        m_Particles[i]->SetPosition(host_position[i]);
    }
}
