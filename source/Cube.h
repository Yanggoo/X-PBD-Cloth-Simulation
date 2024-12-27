#pragma once

#include <memory>
#include <vector>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Actor.h"
#include "Collider.h"

class Scene;

class Cube : public Actor, public Collider
{
public:
	Cube(Scene* scene, float width, float height, float depth);
	~Cube();

	void Update(float deltaTime) override;
	void FixedUpdate(float deltaTime) override;
	void Draw() override;
	void Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation) override;
	glm::vec3 ComputeSDF(glm::vec3 position) override;
	void CreateCubeData(float w, float h, float d, std::vector<float>& vertices, std::vector<unsigned int>& indices);

	glm::vec3 m_Dimensions; // Dimensions: width, height, depth
	glm::vec3 m_Position;
    Scene* m_Scene;
	std::vector<float> vertices;
	std::vector<unsigned int> indices;
	GLuint m_vao;
    GLuint m_vbo;
};
