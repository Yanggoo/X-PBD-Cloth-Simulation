#include "ClothSolverBase.h"
#include <GL/glut.h>
#include <iostream>
#include "Cloth.h"
#include "Particle.h"
#include "Collider.h"

glm::vec3 ClothSolverBase::Mouse2World(int mouseX, int mouseY) {

	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);

	GLdouble modelview[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, modelview);

	GLdouble projection[16];
	glGetDoublev(GL_PROJECTION_MATRIX, projection);

	GLfloat depth;
	glReadPixels(mouseX, viewport[3] - mouseY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

	if (depth == 1.0f) {
		std::cout << "Mouse click did not hit any object!" << std::endl;
		return glm::vec3(10, 10, 10);
	}

	GLdouble worldX, worldY, worldZ;
	if (gluUnProject(
		mouseX, viewport[3] - mouseY, depth,
		modelview, projection, viewport,
		&worldX, &worldY, &worldZ
	)) {
		std::cout << "World Coordinates: ("
			<< worldX << ", " << worldY << ", " << worldZ << ")" << std::endl;
		return glm::vec3(worldX, worldY, worldZ);
	}
	else {
		std::cerr << "gluUnProject failed!" << std::endl;
		return glm::vec3(10, 10, 10);
	}
}


glm::vec3 ClothSolverBase::MouseToWorldZ0(int mouseX, int mouseY, float particleZ)
{
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);

	GLdouble modelview[16], projection[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
	glGetDoublev(GL_PROJECTION_MATRIX, projection);

	GLdouble nearX, nearY, nearZ;
	gluUnProject(
		mouseX, viewport[3] - mouseY, 0.0,  // depth = 0.0
		modelview, projection, viewport,
		&nearX, &nearY, &nearZ
	);
	glm::vec3 nearPoint(nearX, nearY, nearZ);

	GLdouble farX, farY, farZ;
	gluUnProject(
		mouseX, viewport[3] - mouseY, 1.0,  // depth = 1.0
		modelview, projection, viewport,
		&farX, &farY, &farZ
	);
	glm::vec3 farPoint(farX, farY, farZ);


	glm::vec3 rayDirection = glm::normalize(farPoint - nearPoint);
	glm::vec3 rayOrigin = nearPoint;

	if (fabs(rayDirection.z) < 1e-6) {
		std::cerr << "Ray is parallel to the plane!" << std::endl;
		return glm::vec3(0.0f);
	}

	float t = (particleZ - rayOrigin.z) / rayDirection.z;
	glm::vec3 intersectionPoint = rayOrigin + t * rayDirection;

	return intersectionPoint;
}

Particle* ClothSolverBase::GetParticleAtScreenPos(int mouseX, int mouseY)
{
	//TO DO: Implement this function for GPU solver
	std::cerr << "TO DO: Implement this function for GPU solver!" << std::endl;
	return nullptr;
}

void ClothSolverBase::setSelectedParticlePosition(Particle* SelectedParticle)
{
	//TO DO: Implement this function for GPU solver
	std::cerr << "TO DO: Implement this function for GPU solver!" << std::endl;
}
