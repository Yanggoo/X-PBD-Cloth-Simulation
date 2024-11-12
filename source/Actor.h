#pragma once
#include "Transform.h"
class Transform; 
class Actor
{
public:
	Actor() :m_Transform(this) {};
	~Actor()=default;
	virtual void Update(float deltaTime);
	virtual void FixedUpdate(float deltaTime);
	virtual void Draw();
	virtual void Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation);
	Transform m_Transform;
};

