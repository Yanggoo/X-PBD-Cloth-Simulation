#pragma once
#include "Actor.h"
#include "Collider.h"

class Cube : public Actor, public Collider
{
public:
	Cube(float width, float height, float depth);
	~Cube();

	void Update(float deltaTime) override;
	void FixedUpdate(float deltaTime) override;
	void Draw() override;
	void Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation) override;
	glm::vec3 ComputeSDF(glm::vec3 position) override;

	glm::vec3 m_Dimensions; // Dimensions: width, height, depth
	glm::vec3 m_Position;
};
