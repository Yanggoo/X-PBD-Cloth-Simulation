#pragma once
#include "Actor.h"
#include <vector>
#include <memory>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "ClothSolverBase.h"
#include "ClothSolverCPU.h"
#include "ClothSolverGPU.h"

class Application;

class Scene
{
public:
	Scene(Application* application);
	~Scene();
	unsigned int compileShader(const char* source, GLenum type);
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

	unsigned int m_shaderProgram;
	std::shared_ptr<Application> m_application;
};

