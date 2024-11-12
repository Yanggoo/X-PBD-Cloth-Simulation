#pragma once
#include <glm/glm.hpp>
class Collider
{
public:
	Collider();
	~Collider();
	glm::vec3 ComputeSDF(glm::vec3  position);
};

