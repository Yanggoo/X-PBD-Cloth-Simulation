#include "SolverKernel.cuh"

#include <device_launch_parameters.h>


using namespace ClothSolver;


__global__ void kernCalculatePredictPosition(glm::vec3* position, glm::vec3* predictPosition, const float* invMasses, glm::vec3* velocity, float deltaTime) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * blockDim.x * gridDim.x + x;

    if(invMasses[index] == 0.0f) return;
    glm::vec3 pos = position[index];
    velocity[index] += glm::vec3(0, -9.8, 0) * deltaTime;
    predictPosition[index] = pos + velocity[index] * deltaTime;
}

/// <summary>
///  Solve the stretch constraints
/// </summary>
/// <param name="updateDirection"> 0 for horizontal 1 for vertical</param>
/// <returns></returns>
__global__ void kernSolveStretch(
    glm::vec3* predictPositions,
    const glm::vec3* positions,
    const float* invMasses,
    const float constraintsDistances, const int numWidth, const int numHeight,
    int numConstraints,
    float epsilon, int updateDirection)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * blockDim.x * gridDim.x + x;


    int idx1, idx2;

    if (index >= numConstraints) return;
        

    switch (updateDirection)
    {
        case 0: // Horizontal constraint
            if (x + 1 >= numWidth) return; 
            idx1 = index;
            idx2 = index + 1;
            break;

        case 1: // Vertical constraint
            if (y + 1 >= numHeight) return; 
            idx1 = index;
            idx2 = index + numWidth; 
            break;

        case 2: // Horizontal constraint(Even Row)
            if ((y % 2 != 0) || x + 1 >= numWidth) return;
            idx1 = index;
            idx2 = index + 1;
            break;

        case 3: // Vertical constraint(Odd col)
            if ((y % 2 == 0) || y + 1 >= numHeight) return;
            idx1 = index;
            idx2 = index + numWidth;
            break;

        default:
            return; 
    }


    float restDistance = constraintsDistances;




    glm::vec3 p1p2 = predictPositions[idx1] - predictPositions[idx2];
    float currentDistance = glm::length(p1p2);

    if (currentDistance > restDistance) {
        float invMass1 = invMasses[idx1];
        float invMass2 = invMasses[idx2];
        if (invMass1 + invMass2 > 0.0f) {
            float C = currentDistance - restDistance;
            glm::vec3 gradient = p1p2 / (currentDistance + epsilon);
            float deltaLambda = -C / (invMass1 + invMass2);

            if (invMass1 > 0.0f) {
                predictPositions[idx1] += gradient * deltaLambda * invMass1;
            }
            if (invMass2 > 0.0f) {
                predictPositions[idx2] -= gradient * deltaLambda * invMass2;
            }
        }
    }
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

void ClothSolver::CalculatePredictPosition(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, const float* invMasses, glm::vec3* velocity, float deltaTime) {
    kernCalculatePredictPosition << <blocksPerGrid, threadsPerBlock >> > (position, predictPosition, invMasses, velocity, deltaTime);

}

void ClothSolver::UpdateVelocityAndWriteBack(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predictPosition, glm::vec3* velocity, float deltaTime, float damping, int numParticles) {
	updateVelocityAndWriteBack << <blocksPerGrid, threadsPerBlock >> > (position, predictPosition, velocity, deltaTime, damping, numParticles);
}



void ClothSolver::SolveStretchConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predictPositions, const glm::vec3* positions, const float* invMasses, const float constraintsDistances, const int numWidth, const int numHeight, int numConstraints, float epsilon, int updateDirection) {
	kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances, numWidth, numHeight, numConstraints, epsilon, updateDirection);
    cudaDeviceSynchronize();
}



