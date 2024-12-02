#pragma once
#include <vector>
#include "Particle.h"
#include "Collider.h"
#include <tuple>
#include "ClothSolverBase.h"
#include "DynamicKDTreeCPU.h"
#include <cmath>

class Cloth;
class ClothSolverCPU : public ClothSolverBase
{
public:
	ClothSolverCPU();
	~ClothSolverCPU() = default;

	void ResponsibleFor(Cloth* cloth) override;
	void Simulate(float deltaTime) override;
	void PredictPositions(float deltaTime);
	void CollideSDF(std::vector<glm::vec3>& position);
	void SolveStretch(float deltaTime);
	void SolveBending(float deltaTime);
	void SolveParticleCollision();
	void WriteBackPositions();
	//Screen functions
	virtual Particle* GetParticleAtScreenPos(int mouseX, int mouseY)override;
	virtual void setSelectedParticlePosition(Particle* SelectedParticle)override;
	glm::vec3 ComputeFriction(glm::vec3 correction, glm::vec3 relativeVelocity) {
		float length = glm::length(correction);
		if (m_Friction > 0 && length > 0) {
			glm::vec3 correctionDir = correction / length;
			glm::vec3 tangent = relativeVelocity - glm::dot(relativeVelocity, correctionDir) * correctionDir;
			float tangentLength = glm::length(tangent);
			if(tangentLength==0)
				return glm::vec3(0);
			glm::vec3 tangentDir = tangent / tangentLength;
			float maxTangential = length * m_Friction;
			return -tangentDir*std::min(length*m_Friction, tangentLength);
		}
		else {
			return glm::vec3(0);
		}
	}


	std::vector<glm::vec3> m_PredPositions;
	std::vector<glm::vec3> m_Positions;
	std::vector<glm::vec3> m_Velocities;
	std::vector<Particle*> m_Particles;
	//std::vector<Collider*> m_Colliders;
	std::vector<float>m_Lambdas;
	int m_ParticlesNum;
	int m_IterationNum;
	int m_Substeps;
	const float m_BendCompliance = 0.5;
	const float m_Damping = 1;
	const float m_Epsilon = 1e-6;
	const float m_MinDistanceBetweenParticles = 0.01;
	const float m_Friction = 0.1f;
	const float m_MaxVelecity = 10.0f;

	std::vector<std::tuple<int, int, float>> m_StretchConstraints; // idx1, idx2, distance
	std::vector<std::tuple<int, int, int, int, float>> m_BendingConstraints; // idx1, idx2, idx3, idx4, angle
	DynamicKDTreeCPU m_KDTree;

	//std::vector<std::tuple<Particle*, Particle*, Particle*, Particle*>> m_SelfCollisionConstraints; // idx1, triangle(idx2, idx3, idx4)
	//std::vector<std::tuple<Particle*, glm::vec3>> m_AttachmentConstriants; // idx1, position


};

