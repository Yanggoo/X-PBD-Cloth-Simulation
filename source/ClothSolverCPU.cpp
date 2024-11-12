#include "ClothSolverCPU.h"
#include <tuple>

ClothSolverCPU::ClothSolverCPU()
{
	m_Particles.resize(0);
	m_Colliders.resize(0);
	m_PredPositions.resize(0);
	m_Positions.resize(0);
	m_Velocities.resize(0);
	m_StretchConstraints.resize(0);
	m_BendingConstraints.resize(0);
	m_ParticlesNum = 0;
	m_IterationNum = 20;
	m_Substeps = 5;
}

void ClothSolverCPU::ResponsibleFor(Cloth* cloth)
{
	int NumWidth = cloth->m_NumWidth;
	int NumHeight = cloth->m_NumHeight;
	for (int w = 0; w < NumWidth; w++)
	{
		for (int h = 0; h < NumHeight; h++)
		{
			m_Particles.push_back(&cloth->m_Particles[w * NumHeight + h]);
			m_PredPositions.push_back(cloth->m_Particles[w * NumHeight + h].GetPosition());
			m_Positions.push_back(cloth->m_Particles[w * NumHeight + h].GetPosition());
			m_Velocities.push_back(glm::vec3(0.0f));
			m_ParticlesNum++;
			// Stretch Constraints
			if (w < NumWidth - 1)
				m_StretchConstraints.push_back(
					std::make_tuple(&cloth->m_Particles[w * NumHeight + h], 
						&cloth->m_Particles[(w + 1) * NumHeight + h], 
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition() 
							- cloth->m_Particles[(w + 1) * cloth->m_NumHeight + h].GetPosition())));
			if (h < NumHeight - 1)
				m_StretchConstraints.push_back(
					std::make_tuple(&cloth->m_Particles[w * NumHeight + h],
						&cloth->m_Particles[w * NumHeight + h + 1],
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
							- cloth->m_Particles[w * cloth->m_NumHeight + h + 1].GetPosition())));
			// Bending Constraints
			if (w < NumWidth - 1 && h < NumHeight - 1)
			{
				m_BendingConstraints.push_back(
					std::make_tuple(&cloth->m_Particles[w * NumHeight + h],
						&cloth->m_Particles[(w + 1) * NumHeight + h],
						&cloth->m_Particles[w * NumHeight + h + 1],
						&cloth->m_Particles[(w + 1) * NumHeight + h + 1],
						glm::radians(180.0f)));
			}
		}
	}
}

void ClothSolverCPU::Simulate(float deltaTime)
{
	CollideSDF(m_Positions);

	PredictPositions(deltaTime);

	//Solve constrains
	float deltaTimeInSubstep = deltaTime / m_Substeps;
	for (int substep = 0; substep < m_Substeps; substep++) {
		PredictPositions(deltaTimeInSubstep);
		for (int i = 0; i < m_IterationNum; i++)
		{
			SolveStretch(deltaTimeInSubstep);
			SolveBending(deltaTimeInSubstep);
			SolveParticleCollision();
			CollideSDF(m_PredPositions);
		}

		for (int i = 0; i < m_ParticlesNum; i++) {
			m_Velocities[i] = (m_PredPositions[i] - m_Positions[i]) / deltaTimeInSubstep;
			m_Positions[i] = m_PredPositions[i];
		}
	}


	WriteBackPositions();
}

void ClothSolverCPU::PredictPositions(float deltaTime)
{
	for (int i = 0; i < m_ParticlesNum; i++)
	{
		if (m_Particles[i]->m_InvMass == 0.0f) continue;
		glm::vec3 pos = m_Particles[i]->GetPosition();
		m_Velocities[i] += glm::vec3(0,-9.8,0)*deltaTime;
		m_PredPositions[i] = pos + m_Velocities[i] * deltaTime;
	}
}


void ClothSolverCPU::CollideSDF(std::vector<glm::vec3>& position)
{
	for (int i = 0; i < m_ParticlesNum; i++) {
		for (auto col : m_Colliders)
		{
			auto pos = position[i];
			glm::vec3 correction = col->ComputeSDF(pos);
			position[i] += correction;
		}
	}
}

void ClothSolverCPU::SolveStretch(float deltaTime)
{

}

void ClothSolverCPU::SolveBending(float deltaTime)
{

}

void ClothSolverCPU::SolveParticleCollision()
{

}

void ClothSolverCPU::WriteBackPositions()
{
	for (int i = 0; i < m_ParticlesNum; i++) {
		m_Particles[i]->SetPosition(m_Positions[i]);
	}
}
