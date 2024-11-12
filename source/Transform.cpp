#include "Transform.h"
glm::mat4 Transform::matrix()
{
	glm::mat4 result = glm::mat4(1.0f);
	result = glm::translate(result, m_Position);
	// Yaw, Pitch, Roll
	result = glm::rotate(result, glm::radians(m_Rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	result = glm::rotate(result, glm::radians(m_Rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	result = glm::rotate(result, glm::radians(m_Rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
	result = glm::scale(result, m_Scale);
	return result;
}

void Transform::Reset()
{
	m_Position = glm::vec3(0.0f);
	m_Rotation = glm::vec3(0.0f);
	m_Scale = glm::vec3(1.0f);
}