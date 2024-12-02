#include "ClothSolverCPU.h"
#include <tuple>
#include "Cloth.h"

ClothSolverCPU::ClothSolverCPU() :m_ParticlesNum(0), m_IterationNum(10), m_Substeps(5), m_KDTree(m_PredPositions)
{
	m_Particles.resize(0);
	m_Colliders.resize(0);
	m_PredPositions.resize(0);
	m_Positions.resize(0);
	m_Velocities.resize(0);
	m_StretchConstraints.resize(0);
	m_BendingConstraints.resize(0);
	m_ShirnkConstraints.resize(0);
	m_Lambdas.resize(0);
}

void ClothSolverCPU::ResponsibleFor(Cloth* cloth)
{
	int NumWidth = cloth->m_NumWidth;
	int NumHeight = cloth->m_NumHeight;
	int idxOffset = m_ParticlesNum;
	for (int w = 0; w < NumWidth; w++)
	{
		for (int h = 0; h < NumHeight; h++)
		{
			m_Particles.push_back(&cloth->m_Particles[w * NumHeight + h]);
			m_PredPositions.push_back(cloth->m_Particles[w * NumHeight + h].GetPosition());
			m_Positions.push_back(cloth->m_Particles[w * NumHeight + h].GetPosition());
			m_Velocities.push_back(glm::vec3(0.0f));
			m_ParticlesNum++;
			int scale;
			// Stretch Constraints
			scale = 1;
			if (w+scale < NumWidth)
				m_StretchConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						(w + scale) * NumHeight + h + idxOffset,
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
							- cloth->m_Particles[(w + scale) * cloth->m_NumHeight + h].GetPosition())));
			if (h + scale < NumHeight)
				m_StretchConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						w * NumHeight + h + scale + idxOffset,
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
							- cloth->m_Particles[w * cloth->m_NumHeight + h + scale].GetPosition())));
			//Large Scale Stretch Constraints
			scale = 2;
			if (w + scale < NumWidth)
				m_StretchConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						(w + scale) * NumHeight + h + idxOffset,
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
							- cloth->m_Particles[(w + scale) * cloth->m_NumHeight + h].GetPosition())));
			if (h + scale < NumHeight)
				m_StretchConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						w * NumHeight + h + scale + idxOffset,
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
							- cloth->m_Particles[w * cloth->m_NumHeight + h + scale].GetPosition())));
			
			//Large Scale Shrink Constraints
			scale = 4;
			if (w + scale < NumWidth)
			{
				m_ShirnkConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						(w + scale) * NumHeight + h + idxOffset,
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
							- cloth->m_Particles[(w + scale) * cloth->m_NumHeight + h].GetPosition())));
			}
			if (h + scale < NumHeight)
			{
				m_ShirnkConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						w * NumHeight + h + scale + idxOffset,
						glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
							- cloth->m_Particles[w * cloth->m_NumHeight + h + scale].GetPosition())));
			}
			//scale = 8;
			//if (w + scale < NumWidth)
			//{
			//	m_ShirnkConstraints.push_back(
			//		std::make_tuple(w * NumHeight + h + idxOffset,
			//			(w + scale) * NumHeight + h + idxOffset,
			//			glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
			//				- cloth->m_Particles[(w + scale) * cloth->m_NumHeight + h].GetPosition())));
			//}
			//if (h + scale < NumHeight)
			//{
			//	m_ShirnkConstraints.push_back(
			//		std::make_tuple(w * NumHeight + h + idxOffset,
			//			w * NumHeight + h + scale + idxOffset,
			//			glm::length(cloth->m_Particles[w * cloth->m_NumHeight + h].GetPosition()
			//				- cloth->m_Particles[w * cloth->m_NumHeight + h + scale].GetPosition())));
			//}
		
			// Bending Constraints
			scale = 1;
			if (w < NumWidth - 1 && h < NumHeight - 1)
			{
				m_BendingConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						(w + 1) * NumHeight + h + idxOffset,
						w * NumHeight + h + 1 + idxOffset,
						(w + 1) * NumHeight + h + 1 + idxOffset,
						glm::radians(0.0f)));
				m_Lambdas.push_back(0.0f);
			}

			// Large Scale Bending Constraints
			scale = 2;
			if (w + scale < NumWidth && h + scale < NumHeight)
			{
				m_BendingConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						(w + scale) * NumHeight + h + idxOffset,
						w * NumHeight + h + scale + idxOffset,
						(w + scale) * NumHeight + h + scale + idxOffset,
						glm::radians(0.0f)));
				m_Lambdas.push_back(0.0f);
			}

			scale = 4;
			if (w + scale < NumWidth && h + scale < NumHeight)
			{
				m_BendingConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						(w + scale) * NumHeight + h + idxOffset,
						w * NumHeight + h + scale + idxOffset,
						(w + scale) * NumHeight + h + scale + idxOffset,
						glm::radians(0.0f)));
				m_Lambdas.push_back(0.0f);
			}
			scale = 8;
			if (w + scale < NumWidth && h + scale < NumHeight)
			{
				m_BendingConstraints.push_back(
					std::make_tuple(w * NumHeight + h + idxOffset,
						(w + scale) * NumHeight + h + idxOffset,
						w * NumHeight + h + scale + idxOffset,
						(w + scale) * NumHeight + h + scale + idxOffset,
						glm::radians(0.0f)));
				m_Lambdas.push_back(0.0f);
			}
		}
	}
}

void ClothSolverCPU::Simulate(float deltaTime)
{
	CollideSDF(m_Positions);
	//Solve constrains
	float deltaTimeInSubstep = deltaTime / m_Substeps;
	for (int substep = 0; substep < m_Substeps; substep++) {
		PredictPositions(deltaTimeInSubstep);
		m_KDTree.rebuild();
		fill(m_Lambdas.begin(), m_Lambdas.end(), 0.0f);
		for (int i = 0; i < m_IterationNum; i++)
		{
			SolveStretch(deltaTimeInSubstep);
			//SolveShrink(deltaTimeInSubstep);
			SolveBending(deltaTimeInSubstep);
			SolveParticleCollision();
			CollideSDF(m_PredPositions);
		}
		for (int i = 0; i < m_ParticlesNum; i++) {
			m_Velocities[i] = (m_PredPositions[i] - m_Positions[i]) / deltaTimeInSubstep;
			m_Velocities[i] = m_Velocities[i] * glm::clamp((1.0f - m_Damping * deltaTime), 0.0f, 1.0f);
			if (glm::length(m_Velocities[i]) > m_MaxVelecity)
				m_Velocities[i] = glm::normalize(m_Velocities[i]) * m_MaxVelecity;
			m_Positions[i] = m_PredPositions[i];
		}
	}


	WriteBackPositions();
}

void ClothSolverCPU::OnInitFinish() {}

void ClothSolverCPU::PredictPositions(float deltaTime)
{
	for (int i = 0; i < m_ParticlesNum; i++)
	{
		if (m_Particles[i]->m_InvMass == 0.0f) continue;
		glm::vec3 pos = m_Positions[i];
		m_Velocities[i] += glm::vec3(0, -m_Gravity, 0) * deltaTime;
		//m_Velocities[i][0] = glm::clamp(m_Velocities[i][0], -10.f, 10.f);
		//m_Velocities[i][1] = glm::clamp(m_Velocities[i][1], -10.f, 10.f);
		//m_Velocities[i][2] = glm::clamp(m_Velocities[i][2], -10.f, 10.f);

		m_PredPositions[i] = pos + m_Velocities[i] * deltaTime;
	}
}


void ClothSolverCPU::CollideSDF(std::vector<glm::vec3>& position)
{
	for (int i = 0; i < m_ParticlesNum; i++) {
		for (auto col : m_Colliders)
		{
			if (m_Particles[i]->m_InvMass == 0.0f) continue;
			auto pos = position[i];
			glm::vec3 correction = col->ComputeSDF(pos);
			position[i] += correction;
			glm::vec3 relativeVelocity = position[i] - m_Positions[i];
			glm::vec3 friction = ComputeFriction(correction, relativeVelocity);
			position[i] += friction;
		}
	}
}

void ClothSolverCPU::SolveStretch(float deltaTime)
{
	for (auto constrain : m_StretchConstraints) {
		int idx1 = std::get<0>(constrain);
		int idx2 = std::get<1>(constrain);
		Particle* p1 = m_Particles[idx1];
		Particle* p2 = m_Particles[idx2];
		auto w1 = p1->m_InvMass;
		auto w2 = p2->m_InvMass;
		float distance = std::get<2>(constrain);
		glm::vec3 p1p2 = m_PredPositions[idx1] - m_PredPositions[idx2];
		float currentDistance = glm::length(p1p2);
		if (currentDistance > distance && w1 + w2 > 0) {
			// alpha equals to 0, because stiffness is infinite
			float C = currentDistance - distance;
			glm::vec3 gradientP1 = p1p2 / (currentDistance + m_Epsilon);
			glm::vec3 gradientP2 = -p1p2 / (currentDistance + m_Epsilon);
			float deltaLambda = -C / (w1 + w2);//should be /(w1*glm::lenth2(gradientP1)+...) But lenth2(gradientP1) equals to 1
			m_PredPositions[idx1] += gradientP1 * deltaLambda * w1;
			m_PredPositions[idx2] += gradientP2 * deltaLambda * w2;
		}

	}
}

void ClothSolverCPU::SolveShrink(float deltaTime)
{
	for (auto constrain : m_StretchConstraints) {
		int idx1 = std::get<0>(constrain);
		int idx2 = std::get<1>(constrain);
		Particle* p1 = m_Particles[idx1];
		Particle* p2 = m_Particles[idx2];
		auto w1 = p1->m_InvMass;
		auto w2 = p2->m_InvMass;
		float distance = std::get<2>(constrain);
		glm::vec3 p1p2 = m_PredPositions[idx1] - m_PredPositions[idx2];
		float currentDistance = glm::length(p1p2);
		if (currentDistance < distance && w1 + w2 > 0) {
			// alpha equals to 0, because stiffness is infinite
			float C = currentDistance - distance;
			glm::vec3 gradientP1 = p1p2 / (currentDistance + m_Epsilon);
			glm::vec3 gradientP2 = -p1p2 / (currentDistance + m_Epsilon);
			float denominator = w1+ w2;
			float deltaLambda = -C / (w1 + w2);
			m_PredPositions[idx1] += gradientP1 * deltaLambda * w1 * 0.5f;
			m_PredPositions[idx2] += gradientP2 * deltaLambda * w2 * 0.5f;
		}

	}
}

void ClothSolverCPU::SolveBending(float deltaTime)
{
	float alpha = m_BendCompliance / (deltaTime * deltaTime + m_Epsilon);
	for (int i = 0; i < m_BendingConstraints.size(); i++) {
		auto constrain = m_BendingConstraints[i];

		int idx0 = std::get<0>(constrain);
		int idx1 = std::get<1>(constrain);
		int idx2 = std::get<2>(constrain);
		int idx3 = std::get<3>(constrain);
		auto w0 = m_Particles[idx0]->m_InvMass;
		auto w1 = m_Particles[idx1]->m_InvMass;
		auto w2 = m_Particles[idx2]->m_InvMass;
		auto w3 = m_Particles[idx3]->m_InvMass;
		glm::vec3 p0 = m_PredPositions[idx0];
		glm::vec3 p1 = m_PredPositions[idx1];
		glm::vec3 p2 = m_PredPositions[idx2];
		glm::vec3 p3 = m_PredPositions[idx3];
		float angle = std::get<4>(constrain);
		glm::vec3 n1 = glm::cross(p2 - p0, p1 - p2);
		glm::vec3 n2 = glm::cross(p2 - p1, p3 - p2);
		if (glm::length(n1) < m_Epsilon || glm::length(n2) < m_Epsilon) continue;
		n1 = glm::normalize(n1);
		n2 = glm::normalize(n2);
		float d = glm::clamp(glm::dot(n1, n2), -1.0f, 1.0f);
		float currentAngle = glm::acos(d);
		if (abs(currentAngle - angle) < m_Epsilon || isnan(d)) continue;
		if (w0 + w1 + w2 + w3 > 0) {
			float C = currentAngle - angle;
			glm::vec3 gradientP2 = (glm::cross(p1, n2) + d * glm::cross(n1, p1)) / (glm::length(glm::cross(p1, p2)) + m_Epsilon);
			glm::vec3 gradientP3 = (glm::cross(p1, n1) + d * glm::cross(n2, p1)) / (glm::length(glm::cross(p1, p3)) + m_Epsilon);
			glm::vec3 gradientP1 = -(glm::cross(p2, n2) + d * glm::cross(n1, p2)) / glm::length(glm::cross(p1, p2) + m_Epsilon)
				- (glm::cross(p3, n1) + d * glm::cross(n2, p3)) / glm::length(glm::cross(p1, p3) + m_Epsilon);
			glm::vec3 gradientP0 = -gradientP1 - gradientP2 - gradientP3;
			float denominator
				= w0 * glm::dot(gradientP0, gradientP0)
				+ w1 * glm::dot(gradientP1, gradientP1)
				+ w2 * glm::dot(gradientP2, gradientP2)
				+ w3 * glm::dot(gradientP3, gradientP3)
				+ alpha;
			if (denominator
				< m_Epsilon) continue;
			float deltaLambda = (-C - m_Lambdas[i] * alpha) / denominator;
			m_Lambdas[i] += deltaLambda;
			m_PredPositions[idx0] += gradientP0 * deltaLambda * w0;
			m_PredPositions[idx1] += gradientP1 * deltaLambda * w1;
			m_PredPositions[idx2] += gradientP2 * deltaLambda * w2;
			m_PredPositions[idx3] += gradientP3 * deltaLambda * w3;
		}
	}
}

void ClothSolverCPU::SolveParticleCollision()
{
	for (int i = 0; i < m_ParticlesNum; i++)
	{
		auto neighbors = m_KDTree.queryNeighbors(m_PredPositions[i], 8);
		for (auto neighbor : neighbors)
		{
			//already checked
			if (neighbor <= i) 
				continue;

			glm::vec3 p1p2 = m_PredPositions[i] - m_PredPositions[neighbor];
			float currentDistance = glm::length(p1p2);
			auto w1 = m_Particles[i]->m_InvMass;
			auto w2 = m_Particles[neighbor]->m_InvMass;
			if (currentDistance < m_MinDistanceBetweenParticles && w1+w2>0)
			{
				// alpha equals to 0, because stiffness is infinite
				float C = currentDistance - m_MinDistanceBetweenParticles;
				glm::vec3 gradientP1 = p1p2 / (currentDistance + m_Epsilon);
				glm::vec3 gradientP2 = -p1p2 / (currentDistance + m_Epsilon);
				float deltaLambda = -C / (w1 + w2);//should be /(w1*glm::lenth2(gradientP1)+...) But lenth2(gradientP1) equals to 1
				m_PredPositions[i] += gradientP1 * deltaLambda * w1;
				m_PredPositions[neighbor] += gradientP2 * deltaLambda * w2;

				glm::vec3 relativeVelocity = (m_PredPositions[i] - m_Positions[i]) 
					- (m_PredPositions[neighbor] - m_Positions[neighbor]);
				glm::vec3 friction = ComputeFriction(gradientP1 * deltaLambda, relativeVelocity);
				m_PredPositions[i] += friction * w1;
				m_PredPositions[neighbor] -= friction * w2;
			}
		}
	}
}

void ClothSolverCPU::WriteBackPositions()
{
	for (int i = 0; i < m_ParticlesNum; i++) {
		if (m_Particles[i]->m_InvMass == 0.0f) continue;
		m_Particles[i]->SetPosition(m_Positions[i]);
	}
}

Particle* ClothSolverCPU::GetParticleAtScreenPos(int mouseX, int mouseY)
{
	glm::vec3 worldPos = Mouse2World(mouseX, mouseY);
	if (worldPos == glm::vec3(10, 10, 10)) return nullptr;

	float minDistance = 1000000;
	Particle* closestParticle = nullptr;
	for (int i = 0; i < m_ParticlesNum; i++) {
		float distance = glm::length(worldPos - m_Positions[i]);
		if (distance < minDistance) {
			minDistance = distance;
			closestParticle = m_Particles[i];
		}
	}
	return closestParticle;
}

void ClothSolverCPU::setSelectedParticlePosition(Particle* SelectedParticle)
{
	if (SelectedParticle == nullptr) return;
	for (int i = 0; i < m_ParticlesNum; i++)
	{
		if (m_Particles[i] == SelectedParticle)
		{
			m_PredPositions[i] = SelectedParticle->GetPosition();
			m_Positions[i] = SelectedParticle->GetPosition();
		}
	}
}