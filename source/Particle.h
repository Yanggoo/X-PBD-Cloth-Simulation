#pragma once
#include <glm/glm.hpp>
class Particle
{
public:
	Particle() = default;
	Particle(float inv_mass=1.0f, glm::vec3 position=glm::vec3(0,0,0), glm::vec3 normal = glm::vec3(0, 0, 0)) :
        m_InvMass(inv_mass), m_Position(position), m_Normal(normal)
	{}

	~Particle() = default;
    inline const glm::vec3& GetPosition(){ return m_Position; }
	inline void SetPosition(glm::vec3 position) { m_Position = position; }

	glm::vec3	m_Position;
	glm::vec3	m_Normal;
	float		m_InvMass;
};

