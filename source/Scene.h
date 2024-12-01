#pragma once
#include "Actor.h"
#include <vector>
#include <memory>
#include "ClothSolverBase.h"
#include "ClothSolverCPU.h"
#include "ClothSolverGPU.h"

class Scene
{
public:
	Scene();
	~Scene();
	void AddActor(std::shared_ptr<Actor> actor);
	void RemoveActor(std::shared_ptr<Actor> actor);
	void Update(float deltaTime);
	void FixedUpdate(float deltaTime);
	void Draw();
	void LoadSceneSphereAndCloth();
	void LoadSceneClothAndCloth();
	std::vector<std::shared_ptr<ClothSolverBase>> getSolvers() { return m_Solvers; };
private:
	std::vector<std::shared_ptr<Actor>> m_Actors;
	std::vector<std::shared_ptr<ClothSolverBase>> m_Solvers;
};

