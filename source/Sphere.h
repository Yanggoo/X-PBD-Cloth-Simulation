#pragma once

#include <vector>
#include <memory>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Actor.h"
#include "Collider.h"

class Scene;

class Sphere:public Actor, public Collider
{
public:
	Sphere(Scene* scene, float radius);
	~Sphere();
	void Update(float deltaTime) override;
	void FixedUpdate(float deltaTime) override;
	void generateSphere(float radius, int sectorCount, int stackCount, std::vector<float>& vertices, std::vector<unsigned int>& indices);
	void Draw() override;
	void Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation) override;
	glm::vec3 ComputeSDF(glm::vec3 position) override;

	float m_Radius;
	float m_Frequency;
	const float m_Amplitude = 3.0f;
	glm::vec3 m_Position;
	std::vector<float> vertices;
	std::vector<unsigned int> indices;
	GLuint m_vao;
	GLuint m_vbo;
    Scene* m_Scene;
};

