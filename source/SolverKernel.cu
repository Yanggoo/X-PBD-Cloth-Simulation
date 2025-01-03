#include "SolverKernel.cuh"

#include <device_launch_parameters.h>

#define FRICTION 0.1f
#define EPSILON 1e-6f
#define PARTICLE_MIN_DISTANCE 0.02f
#define GRAVITY 30.0f

using namespace ClothSolver;


__device__ float customMin(float a, float b) {
    return a < b ? a : b;
}

__device__ glm::vec3 ComputeFriction(glm::vec3 correction, glm::vec3 relativeVelocity) {
    float length = glm::length(correction);
    if (FRICTION > 0 && length > 0) {
        glm::vec3 correctionDir = correction / length;
        glm::vec3 tangent = relativeVelocity - glm::dot(relativeVelocity, correctionDir) * correctionDir;
        float tangentLength = glm::length(tangent);
        if (tangentLength == 0)
            return glm::vec3(0);
        glm::vec3 tangentDir = tangent / tangentLength;
        float maxTangential = length * FRICTION;
        return -tangentDir * customMin(length * FRICTION, tangentLength);
    }
    else {
        return glm::vec3(0);
    }
}

__global__ void kernCalculatePredictPosition(glm::vec3* position, glm::vec3* predictPosition, const float* invMasses, glm::vec3* velocity, float deltaTime) {
    //int x = blockDim.x * blockIdx.x + threadIdx.x;
    //int y = blockDim.y * blockIdx.y + threadIdx.y;
    //int index = y * blockDim.x * gridDim.x + x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(invMasses[index] == 0.0f) return;
    glm::vec3 pos = position[index];
    velocity[index] += glm::vec3(0, -GRAVITY, 0) * deltaTime;
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
    int updateDirection, bool isShrink)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;


    int index = y * blockDim.x * gridDim.x + x;
    //if (index >= numConstraints) return;

    int idx1 = -1, idx2 = -1;



    //if (updateDirection == 0) { // Horizontal constraint
    //    if ((x % 2 == 0) && (x + 1 < numWidth)) { // Within index range
    //        idx1 = y * numWidth + x;         // Current
    //        idx2 = y * numWidth + (x + 1);   // Right
    //    }
    //}
    //else if (updateDirection == 1) { // Vertical constraint
    //    if ((y % 2 == 0) && (y + 1 < numHeight)) { 
    //        idx1 = y * numWidth + x;         // Current
    //        idx2 = (y + 1) * numWidth + x;   // Under
    //    }
    //}
    //else if (updateDirection == 2) { // Horizontal constraint (Odd Row)
    //    if ((x % 2 != 0) && (x + 1 < numWidth)) {
    //        idx1 = y * numWidth + x;         // Current
    //        idx2 = y * numWidth + (x + 1);   // Right
    //    }
    //}
    //else if (updateDirection == 3) { // Vertical constraint (Odd Column)
    //    if ((y % 2 != 0) && (y + 1 < numHeight)) {
    //        idx1 = y * numWidth + x;         // Current
    //        idx2 = (y + 1) * numWidth + x;   // Under
    //    }
    //}
    //else {
    //    return;
    //}


#pragma region Optimization For Each Threads Call
    if (updateDirection == 0) { // Horizontal constraint
        // Check if x is even and if there is space for two indices
        if (2 * x + 1 <= numWidth - 1) {
            idx1 = y * numWidth + 2 * x;        // Current
            idx2 = y * numWidth + 2 * x + 1;    // Right
        }

    }
    else if (updateDirection == 1) { // Vertical constraint
        // Check if y is even and if there is space for two indices
        if (2 * y + 1 <= numHeight - 1) {
            idx1 = 2 * y * numWidth + x;       // Current
            idx2 = (2 * y + 1) * numWidth + x; // Down
        }

    }
    else if (updateDirection == 2) { // Horizontal constraint (Odd Row)
        // Check if x is odd and if there is space for two indices
        if (2 * x + 2 < numWidth) {
            idx1 = y * numWidth + 2 * x + 1;    // Current
            idx2 = y * numWidth + 2 * x + 2;    // Right
        }
    }
    else if (updateDirection == 3) { // Vertical constraint (Odd Column)
        // Check if y is odd and if there is space for two indices
        if (2 * y + 2 < numHeight) {
            idx1 = (2 * y + 1) * numWidth + x;       // Current
            idx2 = (2 * y + 2) * numWidth + x; // Down
        }

    }
    else if (updateDirection == 4) {
        // Check if x is even and if there is space for two indices
        if (2 * y + 1 <= numHeight - 1 && x + 1 <= numWidth - 1) {
            idx1 = 2 * y * numWidth + x;       // Current
            idx2 = (2 * y + 1) * numWidth + x + 1; // Down
        }
    }
    else if (updateDirection == 5) {
        // Check if y is odd and if there is space for two indices
        if (2 * y + 2 < numHeight && x + 1 <= numWidth - 1) {
            idx1 = (2 * y + 1) * numWidth + x;       // Current
            idx2 = (2 * y + 2) * numWidth + x + 1; // Down
        }
    }
    else if (updateDirection == 6) {
        // Check if x is even and if there is space for two indices
        if (2 * x + 1 <= numWidth - 1 && y + 1 <= numWidth - 1) {
            idx1 = y * numWidth + 2 * x + 1;        // Current
            idx2 = (y + 1) * numWidth + 2 * x;    // Left
        }
    }
    else if (updateDirection == 7) {
        // Check if x is odd and if there is space for two indices
        if (2 * x + 2 < numWidth && y + 1 <= numWidth - 1) {
            idx1 = y * numWidth + 2 * x + 2;    // Current
            idx2 = (y + 1) * numWidth + 2 * x + 1;    // Right
        }
    }
    else {
        return; // Invalid update direction
    }
#pragma endregion





    //if (idx1 < 0 || idx1 >= numConstraints || idx2 < 0 || idx2 >= numConstraints) 
    //    return;

    if (idx1 < 0 || idx2 < 0 || x + y >= numWidth + numHeight/2)
        return;

    float restDistance = constraintsDistances;




    glm::vec3 p1p2 = predictPositions[idx1] - predictPositions[idx2];
    float currentDistance = glm::length(p1p2);

    float factor = isShrink ? 0.01f : 1.0f;

    if (currentDistance > restDistance && !isShrink || currentDistance < restDistance && isShrink) {
        float invMass1 = invMasses[idx1];
        float invMass2 = invMasses[idx2];
        if (invMass1 + invMass2 > 0.0f) {
            float C = currentDistance - restDistance;
            glm::vec3 gradient = p1p2 / (currentDistance + EPSILON);
            float deltaLambda = -C / (invMass1 + invMass2);

            float factor = isShrink ? 0.01f : 1.0f;
            if (invMass1 > 0.0f) {
                predictPositions[idx1] += gradient * deltaLambda * invMass1 * factor;
            }
            if (invMass2 > 0.0f) {
                predictPositions[idx2] -= gradient * deltaLambda * invMass2 * factor;
            }
        }
    }
}


__global__ void kernSolveBendingConstraints(
    glm::vec3* predPositions,     // Predicted positions of particles
    const float* invMasses,       // Inverse masses of particles
    const int numWidth,           // Number of particles along width
    const int numHeight,          // Number of particles along height
    const float constraintAngle,
    const float compliance,       // Compliance parameter
    float alpha,
    int pos)                // Small value to prevent division by zero
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= numWidth || y >= numHeight)
    {
        return;
    }

    int index = y * blockDim.x * gridDim.x + x;

    // Each thread works on a single constraint
    int totalConstraints = (numWidth + 1) * (numHeight - 1); // Total number of constraints
    if (index >= totalConstraints) return;

    int idx0 = -1, idx1 = -1, idx2 = -1, idx3 = -1;

    if (pos == 0)
    {
        idx0 = 2 * (y * numWidth + x);
        idx1 = idx0 + 1;
        idx2 = idx0 + numWidth;
        idx3 = idx2 + 1;
    }
    else if (pos == 1 && 2 * x + 2 < numWidth)
    {
        idx0 = 2 * (y * numWidth + x) + 1;
        idx1 = idx0 + 1;
        idx2 = idx0 + numWidth;
        idx3 = idx2 + 1;
    }
    else if (pos == 2 && 2 * y + 2 < numHeight)
    {
        idx0 = 2 * (y * numWidth + x) + numWidth;
        idx1 = idx0 + 1;
        idx2 = idx0 + numWidth;
        idx3 = idx2 + 1;
    }
    else if (pos == 3 && 2 * y + 2 < numHeight && 2 * x + 2 < numWidth) {
        idx0 = 2 * (y * numWidth + x) + 1 + numWidth;
        idx1 = idx0 + 1;
        idx2 = idx0 + numWidth;
        idx3 = idx2 + 1;
    }

    if (idx0 < 0 || idx1 < 0 || idx2 < 0 || idx3 < 0 || idx3 > totalConstraints)
    {
        return;
    }



    

    // Fetch positions and inverse masses
    glm::vec3 p0 = predPositions[idx0];
    glm::vec3 p1 = predPositions[idx1] - p0;
    glm::vec3 p2 = predPositions[idx2] - p0;
    glm::vec3 p3 = predPositions[idx3] - p0;

    float w0 = invMasses[idx0];
    float w1 = invMasses[idx1];
    float w2 = invMasses[idx2];
    float w3 = invMasses[idx3];

    // Compute normals for the bending plane
    glm::vec3 n1 = glm::normalize(glm::cross(p1, p2));
    glm::vec3 n2 = glm::normalize(glm::cross(p1, p3));

    if (glm::length(n1) < EPSILON || glm::length(n2) < EPSILON) return;

    // Compute the dot product and the current bending angle
    float d = glm::clamp(glm::dot(n1, n2), -1.0f, 1.0f);
    float currentAngle = glm::acos(d);



    if (w0 + w1 + w2 + w3 <= 0.0f) return;

    // Compute constraint force and gradients
    float C = currentAngle - constraintAngle;

    if (fabs(C) < EPSILON || isnan(C)) return;

    glm::vec3 gradientP2 = (glm::cross(p1, n2) + d * glm::cross(n1, p1)) / (glm::length(glm::cross(p1, p2)) + EPSILON);
    glm::vec3 gradientP3 = (glm::cross(p1, n1) + d * glm::cross(n2, p1)) / (glm::length(glm::cross(p1, p3)) + EPSILON);
    glm::vec3 gradientP1 = -(glm::cross(p2, n2) + d * glm::cross(n1, p2)) / (glm::length(glm::cross(p1, p2)) + EPSILON)
        - (glm::cross(p3, n1) + d * glm::cross(n2, p3)) / (glm::length(glm::cross(p1, p3)) + EPSILON);
    glm::vec3 gradientP0 = -gradientP1 - gradientP2 - gradientP3;


    float denominator = w0 * glm::dot(gradientP0, gradientP0)
        + w1 * glm::dot(gradientP1, gradientP1)
        + w2 * glm::dot(gradientP2, gradientP2)
        + w3 * glm::dot(gradientP3, gradientP3)
        + alpha;

    if (denominator < EPSILON) return;

    // Compute delta lambda
    float deltaLambda = glm::sqrt(1.0f - d*d) * C / denominator;

    // Update predicted positions
    predPositions[idx0] += deltaLambda * w0 * gradientP0;
    predPositions[idx1] += deltaLambda * w1 * gradientP1;
    predPositions[idx2] += deltaLambda * w2 * gradientP2;
    predPositions[idx3] += deltaLambda * w3 * gradientP3;
}


__global__ void kernSolveCollisionSphere(glm::vec3* predPositions, glm::vec3* positions, const float* invMasses, 
    glm::vec3 center, float radius)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * blockDim.x * gridDim.x + x;

    if (invMasses[index] == 0.0f) {
        return;
    }

    glm::vec3& position = predPositions[index];
    glm::vec3 offset;
    if (glm::length(position - center) < 0.1f + radius) {
        offset = glm::normalize(position - center) * (0.1f + radius - glm::length(position - center));
    }
    else {
        offset = glm::vec3(0.0f);
    }

    position += offset;

    glm::vec3 relativeVelocity = position - positions[index];
    glm::vec3 friction = ComputeFriction(offset, relativeVelocity);
    position += friction;
}

__global__ void kernSolveCollisionCube(glm::vec3* predPositions, glm::vec3* positions, const float* invMasses, glm::vec3 center, glm::vec3 dimensions)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * blockDim.x * gridDim.x + x;

    if (invMasses[index] == 0.0f) {
        return;
    }

    glm::vec3& position = predPositions[index];
    glm::vec3 offset;
    glm::vec3 halfExtents = dimensions * 0.5f + 0.1f;
    glm::vec3 diff = position - center;
    if (diff.x > -halfExtents.x && diff.x<halfExtents.x
        && diff.y>-halfExtents.y && diff.y < halfExtents.y
        && diff.z>-halfExtents.z && diff.z < halfExtents.z) {
        float dx = diff.x > 0 ? halfExtents.x - diff.x : -halfExtents.x - diff.x;
        float dy = diff.y > 0 ? halfExtents.y - diff.y : -halfExtents.y - diff.y;
        float dz = diff.z > 0 ? halfExtents.z - diff.z : -halfExtents.z - diff.z;
        if (abs(dx) <= abs(dy) && abs(dx) <= abs(dz)) {
            offset = glm::vec3(dx, 0.0f, 0.0f);
        }
        else if (abs(dy) <= abs(dx) && abs(dy) <= abs(dz)) {
            offset = glm::vec3(0.0f, dy, 0.0f);
        }
        else {
            offset = glm::vec3(0.0f, 0.0f, dz);
        }
    }
    else {
        offset = glm::vec3(0.0f);
    }
    
    position += offset;

    glm::vec3 relativeVelocity = position - positions[index];
    glm::vec3 friction = ComputeFriction(offset, relativeVelocity);
    position += friction;
}

__global__ void kernSolveCollisionParticle(glm::vec3* position, glm::vec3* predPosition1, glm::vec3* predPosition2, const float* invMasses, const int* neighbors, const int particleCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //auto neighbors = m_KDTree.queryNeighbors(m_PredPositions[i], 8);
    //for (int neighbor = 0; neighbor < particleCount; ++neighbor) {
    for (int i = 0; i < 8; ++i) {
        int neighbor = neighbors[index * 8 + i];
        //already checked
        if (neighbor == index)
            continue;

        glm::vec3 p1p2 = predPosition1[index] - predPosition1[neighbor];
        float currentDistance = glm::length(p1p2);
        auto w1 = invMasses[index];
        auto w2 = invMasses[neighbor];
        if (currentDistance < PARTICLE_MIN_DISTANCE && w1 + w2>0) {
            // alpha equals to 0, because stiffness is infinite
            float C = currentDistance - PARTICLE_MIN_DISTANCE;
            glm::vec3 gradientP1 = p1p2 / (currentDistance + EPSILON);
            glm::vec3 gradientP2 = -p1p2 / (currentDistance + EPSILON);
            float deltaLambda = -C / (w1 + w2);//should be /(w1*glm::lenth2(gradientP1)+...) But lenth2(gradientP1) equals to 1
            predPosition2[index] += gradientP1 * deltaLambda * w1;
            //m_PredPositions[neighbor] += gradientP2 * deltaLambda * w2;

            glm::vec3 relativeVelocity = (predPosition1[index] - position[index])
                - (predPosition1[neighbor] - position[neighbor]);
            glm::vec3 friction = ComputeFriction(gradientP1 * deltaLambda, relativeVelocity);
            predPosition2[index] += friction * w1;
            //m_PredPositions[neighbor] -= friction * w2;
            //glm::vec3 neighborRelativeVelocity = (predPosition1[neighbor] - position[neighbor]) - (predPosition1[index] - position[index]);
            //glm::vec3 neighborFriction = ComputeFriction(gradientP1 * deltaLambda, neighborRelativeVelocity);
            //predPosition2[index] -= neighborFriction * w1;


            glm::vec3 p1p2Inv = predPosition1[neighbor] - predPosition1[index];
            glm::vec3 gradientP1Inv = p1p2Inv / (currentDistance + EPSILON);
            glm::vec3 gradientP2Inv = -p1p2Inv / (currentDistance + EPSILON);
            predPosition2[index] += gradientP2Inv * deltaLambda * w1;

            glm::vec3 neighborRelativeVelocity = (predPosition1[neighbor] - position[neighbor]) - (predPosition1[index] - position[index]);
            glm::vec3 neighborFriction = ComputeFriction(gradientP2Inv * deltaLambda, neighborRelativeVelocity);
            predPosition2[index] -= neighborFriction * w1;
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
    //int x = blockDim.x * blockIdx.x + threadIdx.x;
    //int y = blockDim.y * blockIdx.y + threadIdx.y;
    //int idx = y * blockDim.x * gridDim.x + x;
    ////int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx >= numParticles) return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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


void ClothSolver::SolveStretchConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predictPositions, const glm::vec3* positions, const float* invMasses, const float* constraintsDistances, const int numWidth, const int numHeight) {
	kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 0, false);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 1, false);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 2, false);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 3, false);
    cudaDeviceSynchronize();

    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 4, false);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 5, false);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 6, false);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 7, false);
    cudaDeviceSynchronize();


    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 0, true);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 1, true);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 2, true);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[0], numWidth, numHeight, 3, true);
    cudaDeviceSynchronize();

    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 4, true);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 5, true);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 6, true);
    cudaDeviceSynchronize();
    kernSolveStretch << <blocksPerGrid, threadsPerBlock >> > (predictPositions, positions, invMasses, constraintsDistances[1], numWidth, numHeight, 7, true);
    cudaDeviceSynchronize();
}

void ClothSolver::SolveBendingConstraints(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, const float* invMasses, const int numWidth, const int numHeight, const float constraintDistance, const float compliance, float alpha) {
	kernSolveBendingConstraints << <blocksPerGrid, threadsPerBlock >> > (predPositions, invMasses, numWidth, numHeight, constraintDistance, compliance, alpha, 0);
    cudaDeviceSynchronize();

    kernSolveBendingConstraints << <blocksPerGrid, threadsPerBlock >> > (predPositions, invMasses, numWidth, numHeight, constraintDistance, compliance, alpha, 1);
    cudaDeviceSynchronize();

    kernSolveBendingConstraints << <blocksPerGrid, threadsPerBlock >> > (predPositions, invMasses, numWidth, numHeight, constraintDistance, compliance, alpha, 2);
    cudaDeviceSynchronize();

    kernSolveBendingConstraints << <blocksPerGrid, threadsPerBlock >> > (predPositions, invMasses, numWidth, numHeight, constraintDistance, compliance, alpha, 3);
	cudaDeviceSynchronize();
}

void ClothSolver::SolveCollisionSphere(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, glm::vec3* positions, 
    const float* invMasses, glm::vec3 center, float radius)
{
    kernSolveCollisionSphere << <blocksPerGrid, threadsPerBlock >> > (predPositions, positions, invMasses, center, radius);
}

void ClothSolver::SolveCollisionCube(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* predPositions, glm::vec3* positions, const float* invMasses, glm::vec3 center, glm::vec3 dimensions) {
    kernSolveCollisionCube << <blocksPerGrid, threadsPerBlock >> > (predPositions, positions, invMasses, center, dimensions);
}

void ClothSolver::SolveCollisionParticle(dim3 blocksPerGrid, dim3 threadsPerBlock, glm::vec3* position, glm::vec3* predPosition1, glm::vec3* predPosition2, const float* invMasses, const int* neighbors, const int particleCount) {
    kernSolveCollisionParticle << <blocksPerGrid, threadsPerBlock >> > (position, predPosition1, predPosition2, invMasses, neighbors, particleCount);
}
