#pragma once
#include "Actor.h"
#include "Collider.h"
class Sphere:public Actor, public Collider
{
public:
	Sphere(float radius);
	~Sphere();
	void Update(float deltaTime) override;
	void FixedUpdate(float deltaTime) override;
	void Draw() override;
	void Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation) override;
	glm::vec3 ComputeSDF(glm::vec3 position) override;

	float m_Radius;
	float m_Frequency;
	const float m_Amplitude = 2.0f;
	glm::vec3 m_Position;
};

