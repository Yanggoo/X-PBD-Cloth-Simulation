#pragma once

#include <vector>
#include <glm/glm.hpp>
//#include "Collider.h"

class Cloth;
class Collider;
class Particle;

class ClothSolverBase {
public:
    virtual void ResponsibleFor(Cloth* cloth) = 0;
    virtual void OnInitFinish() = 0;
    virtual void Simulate(float deltaTime) = 0;
    virtual glm::vec3 Mouse2World(int mouseX, int mouseY);
    virtual glm::vec3 MouseToWorldZ0(int mouseX, int mouseY, float particleZ);
	virtual Particle* GetParticleAtScreenPos(int mouseX, int mouseY);
	virtual void setSelectedParticlePosition(Particle* SelectedParticle);
    virtual void OnInputSelectParticle(Particle* SelectedParticle);
    virtual void OnInputClearParticle(Particle* SelectedParticle);
    std::vector<Collider*> m_Colliders;
};

