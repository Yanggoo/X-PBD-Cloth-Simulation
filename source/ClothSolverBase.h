#pragma once

#include <vector>
//#include "Collider.h"

class Cloth;
class Collider;

class ClothSolverBase {
public:
    virtual void ResponsibleFor(Cloth* cloth) = 0;
    virtual void Simulate(float deltaTime) = 0;

    std::vector<Collider*> m_Colliders;
};

