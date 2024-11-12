#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
class Actor;
class Transform
{
public:
	Transform(Actor* actor)
	{
		m_Actor = actor;
	}

	~Transform()
	{

	}

	glm::mat4 matrix();

	inline Actor* actor()
	{
		return m_Actor;
	}

	void Reset();

	glm::vec3 m_Position = glm::vec3(0.0f);
	glm::vec3 m_Rotation = glm::vec3(0.0f);
	glm::vec3 m_Scale = glm::vec3(1.0f);
	Actor* m_Actor;
};

