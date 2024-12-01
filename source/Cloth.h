#pragma once
#include "Actor.h"
#include "Particle.h"
#include <vector>
#include "ClothSolverBase.h"

class Cloth :
    public Actor
{
	friend class ClothSolverCPU;
	friend class ClothSolverGPU;
public:
	Cloth(float width, float height, int num_width, int num_height, bool fixed = false);
	~Cloth();
	void Update(float deltaTime) override;
	void FixedUpdate(float deltaTime) override;
	void Draw() override;
	void Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation) override;
	void DrawTriangle(Particle* p1, Particle* p2, Particle* p3, const glm::vec3 color);
	Particle* GetParticle(int w, int h);
	void AddSolver(ClothSolverBase* solver);

private:
	int m_NumWidth;
	int m_NumHeight;
	float m_Width;
	float m_Height;
	bool m_Fixed = true;
	std::vector<Particle> m_Particles;
	ClothSolverBase* m_ClothSolver;
};

