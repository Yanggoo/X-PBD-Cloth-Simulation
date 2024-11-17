#pragma once
#include <vector>
#include "Particle.h"
#include "Collider.h"
#include <tuple>
class Cloth;
class ClothSolverCPU
{
public:
	ClothSolverCPU();
	~ClothSolverCPU() = default;

	void ResponsibleFor(Cloth* cloth);
	void Simulate(float deltaTime);
	void PredictPositions(float deltaTime);
	void CollideSDF(std::vector<glm::vec3>& position);
	void SolveStretch(float deltaTime);
	void SolveBending(float deltaTime);
	void SolveParticleCollision();
	void WriteBackPositions();


	std::vector<glm::vec3> m_PredPositions;
	std::vector<glm::vec3> m_Positions;
	std::vector<glm::vec3> m_Velocities;
	std::vector<Particle*> m_Particles;
	std::vector<Collider*> m_Colliders;
	int m_ParticlesNum;
	int m_IterationNum;
	int m_Substeps;
	const float m_BendCompliance = 10;
	const float m_Damping = 0.1;
	const float m_Epsilon = 1e-6;

	std::vector<std::tuple<int, int, float>> m_StretchConstraints; // idx1, idx2, distance
	std::vector<std::tuple<int, int, int, int, float>> m_BendingConstraints; // idx1, idx2, idx3, idx4, angle

	//std::vector<std::tuple<Particle*, Particle*, Particle*, Particle*>> m_SelfCollisionConstraints; // idx1, triangle(idx2, idx3, idx4)
	//std::vector<std::tuple<Particle*, glm::vec3>> m_AttachmentConstriants; // idx1, position


};

