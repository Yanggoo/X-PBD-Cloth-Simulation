#include "Cube.h"
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

Cube::Cube(float width, float height, float depth)
	: m_Dimensions(width, height, depth), m_Position(glm::vec3(0.0f)) {
}

Cube::~Cube() = default;

void Cube::Update(float deltaTime)
{

}

void Cube::FixedUpdate(float deltaTime)
{

}

// Draw method
void Cube::Draw()
{
	glPushMatrix();
	glTranslatef(m_Position.x, m_Position.y, m_Position.z);

	// Set color
	static const glm::vec3 color(1.0f, 0.0f, 0.0f);
	glColor3fv(glm::value_ptr(color));

	// Scale and draw a unit cube
	glScalef(m_Dimensions.x, m_Dimensions.y, m_Dimensions.z);
	glutSolidCube(1.0f);

	glPopMatrix();
}

void Cube::Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation)
{
	Actor::Initialize(position, scale, rotation);
	m_Position = position;
	m_Dimensions = glm::vec3(m_Dimensions.x * scale.x, m_Dimensions.y * scale.y, m_Dimensions.z * scale.z);
}

glm::vec3 Cube::ComputeSDF(glm::vec3 position)
{
	glm::vec3 halfExtents = m_Dimensions * 0.5f;
	glm::vec3 diff = position - m_Position;
	if (diff.x > -halfExtents.x && diff.x<halfExtents.x
		&& diff.y>-halfExtents.y && diff.y < halfExtents.y
		&& diff.z>-halfExtents.z && diff.z < halfExtents.z) {
		float dx = diff.x > 0 ? halfExtents.x  - diff.x : -halfExtents.x  - diff.x;
		float dy = diff.y > 0 ? halfExtents.y  - diff.y : -halfExtents.y  - diff.y;
		float dz = diff.z > 0 ? halfExtents.z  - diff.z : -halfExtents.z  - diff.z;
		if (abs(dx) <= abs(dy) && abs(dx) <= abs(dz)) {
			return glm::vec3(dx, 0.0f, 0.0f);
		}
		else if (abs(dy) <= abs(dx) && abs(dy) <= abs(dz)) {
			return glm::vec3(0.0f, dy, 0.0f);
		}
		else {
			return glm::vec3(0.0f, 0.0f, dz);
		}
	}
	return glm::vec3(0.0f);

}

