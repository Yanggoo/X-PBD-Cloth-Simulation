#include "Actor.h"

void Actor::Update(float deltaTime)
{

}

void Actor::FixedUpdate(float deltaTime)
{

}

void Actor::Draw()
{

}

void Actor::Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation)
{

	m_Transform.m_Position = position;
	m_Transform.m_Rotation = rotation;
	m_Transform.m_Scale = scale;
}
