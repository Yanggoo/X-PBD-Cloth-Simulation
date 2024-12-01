#include "Sphere.h"
#include <GL/freeglut.h>
Sphere::Sphere(float radius)
{
	m_Frequency = 0;
	m_Position = glm::vec3(0.0f);
	m_Radius = radius;
}

Sphere::~Sphere()
{

}

void Sphere::Update(float deltaTime)
{

}

void Sphere::FixedUpdate(float deltaTime)
{
	m_Frequency += deltaTime;
	if (m_Frequency > 2 * glm::pi<float>())
	{
		m_Frequency -= 2 * glm::pi<float>();
	}
	m_Position = m_Transform.m_Position + glm::vec3(0.0f, 0.0f, -m_Amplitude * glm::sin(m_Frequency));
}

void Sphere::Draw()
{
	glPushMatrix();
	glTranslatef(m_Position.x, m_Position.y, m_Position.z);
	static const glm::vec3 color(0.0f, 0.0f, 1.0f);
	glColor3fv((GLfloat*)&color);
	glutSolidSphere(m_Radius, 20, 20);
	glPopMatrix();
}

void Sphere::Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation)
{
	Actor::Initialize(position, scale, rotation);
	m_Position = position;
	m_Radius = m_Radius * scale.x;
}

glm::vec3 Sphere::ComputeSDF(glm::vec3 position)
{
	if (glm::length(position - m_Position) < 0.1f+m_Radius)
	{
		return glm::normalize(position - m_Position) * (0.1f+m_Radius - glm::length(position - m_Position));
	}
	else
	{
		return glm::vec3(0.0f);
	}
}
