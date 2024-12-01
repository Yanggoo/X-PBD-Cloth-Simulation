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
    velocity[index] = velocity[index] * glm::clamp((1.0f - deltaTime), 0.0f, 1.0f);
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
    //if (x >= numWidth || y >= numHeight) return;


    int index = y * blockDim.x * gridDim.x + x;
    if (index >= numConstraints) return;

    int idx1 = -1, idx2 = -1;
    //int idx1, idx2;


        

    //switch (updateDirection)
    //{
    //    //case 0: // Horizontal constraint
    //    //    if (x + 1 >= numWidth) return; 
    //    //    idx1 = index;
    //    //    idx2 = index + 1;
    //    //    break;

    //    //case 1: // Vertical constraint
    //    //    if (y + 1 >= numHeight) return; 
    //    //    idx1 = index;
    //    //    idx2 = index + numWidth; 

    //    //    break;

    //    //case 2: // Horizontal constraint(Even Row)
    //    //    if ((y % 2 != 0) || x + 1 >= numWidth) return;
    //    //    idx1 = index;
    //    //    idx2 = index + 1;
    //    //    break;

    //    //case 3: // Vertical constraint(Odd col)
    //    //    if ((y % 2 == 0) || y + 1 >= numHeight) return;
    //    //    idx1 = index;
    //    //    idx2 = index + numWidth;
    //    //    break;

    //    //default:
    //    //    return; 



    //    case 0: // Horizontal constraint
    //        if ((x % 2 == 0) && (x + 1 < numWidth)) { // 确保右侧索引合法
    //            idx1 = y * numWidth + x;         // 当前点
    //            idx2 = y * numWidth + (x + 1);   // 右侧点
    //        }
    //        break;

    //    case 1: // Vertical constraint
    //        if ((x % 2 != 0) && (y + 1 < numHeight)) { // 确保下方索引合法
    //            idx1 = y * numWidth + x;         // 当前点
    //            idx2 = (y + 1)  * numWidth + x;   // 下方点
    //        }
    //        break;

    //    case 2: // Horizontal constraint (Even Row)
    //        if ((x % 2 != 0) && (x + 1 < numWidth)) {
    //            idx1 = y * numWidth + x;         // 当前点
    //            idx2 = y * numWidth + (x + 1);   // 右侧点
    //        }
    //        break;

    //    case 3: // Vertical constraint (Odd Column)
    //        if ((x % 2 == 0) && (y + 1 < numHeight)) {
    //            idx1 = y * numWidth + x;         // 当前点
    //            idx2 = (y + 1) * numWidth + x;   // 下方点
    //        }
    //        break;

    //    default:
    //        return; // 无效方向
    //}


    if (updateDirection == 0) { // Horizontal constraint
        if ((x % 2 == 0) && (x + 1 < numWidth)) { // 确保右侧索引合法
            idx1 = y * numWidth + x;         // 当前点
            idx2 = y * numWidth + (x + 1);   // 右侧点
        }
    }
    else if (updateDirection == 1) { // Vertical constraint
        if ((x % 2 != 0) && (y + 1 < numHeight)) { // 确保下方索引合法
            idx1 = y * numWidth + x;         // 当前点
            idx2 = (y + 1) * numWidth + x;   // 下方点
        }
    }
    else if (updateDirection == 2) { // Horizontal constraint (Even Row)
        if ((x % 2 != 0) && (x + 1 < numWidth)) {
            idx1 = y * numWidth + x;         // 当前点
            idx2 = y * numWidth + (x + 1);   // 右侧点
        }
    }
    else if (updateDirection == 3) { // Vertical constraint (Odd Column)
        if ((x % 2 == 0) && (y + 1 < numHeight)) {
            idx1 = y * numWidth + x;         // 当前点
            idx2 = (y + 1) * numWidth + x;   // 下方点
        }
    }
    else {
        return; // 无效方向
    }

    if (idx1 < 0 || idx1 >= numConstraints || idx2 < 0 || idx2 >= numConstraints) return;

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


__global__ void kernSolveBendingConstraints(
    glm::vec3* predPositions,     // Predicted positions of particles
    const float* invMasses,       // Inverse masses of particles
    const int numWidth,           // Number of particles along width
    const int numHeight,          // Number of particles along height
    const float constraintDistance, // Distance between constrained points
    const float compliance,       // Compliance parameter
    float alpha,              
    float epsilon)                // Small value to prevent division by zero
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread works on a single constraint
    int totalConstraints = (numWidth - 1) * (numHeight - 1); // Total number of constraints
    if (index >= totalConstraints) return;

    // Determine the row and column of the current thread
    int row = index / (numWidth - 1); // Current row
    int col = index % (numWidth - 1); // Current column

    // Four particles involved in the bending constraint
    int idx0, idx1, idx2, idx3;

    switch (index % 2) {
    case 0: // Horizontal constraint
        idx0 = row * numWidth + col;        // Top-left
        idx1 = row * numWidth + col + 1;    // Top-right
        idx2 = (row + 1) * numWidth + col;  // Bottom-left
        idx3 = (row + 1) * numWidth + col + 1; // Bottom-right
        break;

    case 1: // Vertical constraint
        idx0 = (row - 1) * numWidth + col + 1; // Top
        idx1 = row * numWidth + col;          // Bottom-left
        idx2 = row * numWidth + col + 1;      // Bottom-right
        idx3 = (row + 1) * numWidth + col;    // Bottom
        break;

    default:
        return;
    }

    // Fetch positions and inverse masses
    glm::vec3 p0 = predPositions[idx0];
    glm::vec3 p1 = predPositions[idx1];
    glm::vec3 p2 = predPositions[idx2];
    glm::vec3 p3 = predPositions[idx3];

    float w0 = invMasses[idx0];
    float w1 = invMasses[idx1];
    float w2 = invMasses[idx2];
    float w3 = invMasses[idx3];

    // Compute normals for the bending plane
    glm::vec3 n1 = glm::normalize(glm::cross(p2 - p0, p1 - p2));
    glm::vec3 n2 = glm::normalize(glm::cross(p2 - p1, p3 - p2));

    // Compute the dot product and the current bending angle
    float d = glm::clamp(glm::dot(n1, n2), -1.0f, 1.0f);
    float currentAngle = glm::acos(d);

    // Compute constraint force and gradients
    float C = currentAngle - constraintDistance;
    if (fabs(C) < epsilon || isnan(C)) return;

    glm::vec3 gradientP2 = (glm::cross(p1, n2) + d * glm::cross(n1, p1)) / (glm::length(glm::cross(p1, p2)) + epsilon);
    glm::vec3 gradientP3 = (glm::cross(p1, n1) + d * glm::cross(n2, p1)) / (glm::length(glm::cross(p1, p3)) + epsilon);
    glm::vec3 gradientP1 = -(glm::cross(p2, n2) + d * glm::cross(n1, p2)) / (glm::length(glm::cross(p1, p2)) + epsilon)
        - (glm::cross(p3, n1) + d * glm::cross(n2, p3)) / (glm::length(glm::cross(p1, p3)) + epsilon);
    glm::vec3 gradientP0 = -gradientP1 - gradientP2 - gradientP3;


    float denominator = w0 * glm::dot(gradientP0, gradientP0)
        + w1 * glm::dot(gradientP1, gradientP1)
        + w2 * glm::dot(gradientP2, gradientP2)
        + w3 * glm::dot(gradientP3, gradientP3)
        + alpha;

    if (denominator < epsilon) return;

    // Compute delta lambda
    float deltaLambda = -C / denominator;

    // Update predicted positions
    if (w0 > 0) predPositions[idx0] += deltaLambda * w0 * gradientP0;
    if (w1 > 0) predPositions[idx1] += deltaLambda * w1 * gradientP1;
    if (w2 > 0) predPositions[idx2] += deltaLambda * w2 * gradientP2;
    if (w3 > 0) predPositions[idx3] += deltaLambda * w3 * gradientP3;
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
    //cudaDeviceSynchronize();
}

void ClothSolver::SolveBendingConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, const float* invMasses, const int numWidth, const int numHeight, const float constraintDistance, const float compliance, float alpha, float epsilon) {
	kernSolveBendingConstraints << <blocksPerGrid, threadsPerBlock >> > (predPositions, invMasses, numWidth, numHeight, constraintDistance, compliance, alpha, epsilon);
	//cudaDeviceSynchronize();
}



