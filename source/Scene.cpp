#include "Scene.h"
#include "Cloth.h"
#include "Sphere.h"
#include "Globals.h"

Scene::Scene()
{

}

Scene::~Scene()
{

}

void Scene::AddActor(std::shared_ptr<Actor>actor)
{
	m_Actors.push_back(actor);
}

void Scene::RemoveActor(std::shared_ptr<Actor> actor)
{
	m_Actors.erase(std::remove(m_Actors.begin(), m_Actors.end(), actor), m_Actors.end());
}

void Scene::Update(float deltaTime)
{
	for (auto actor : m_Actors)
	{
		actor->Update(deltaTime);
	}
}

void Scene::FixedUpdate(float deltaTime)
{
	for (auto actor : m_Actors)
	{
		actor->FixedUpdate(deltaTime);
	}
	for (auto solver : m_Solvers)
	{
		solver.get()->Simulate(deltaTime);
	}
}

void Scene::Draw()
{
	for (auto actor : m_Actors)
	{
		actor->Draw();
	}
}

void Scene::LoadScene()
{
	std::shared_ptr<Sphere> sphere = std::make_shared<Sphere>(0.1f);
	sphere->Initialize(glm::vec3(0, 0, 1), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(sphere);
	std::shared_ptr<Cloth> cloth = std::make_shared<Cloth>(2, 2, 64, 64);
	cloth->Initialize(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(cloth);

	std::shared_ptr<ClothSolverBase> solver;
#if USE_GPU_SOLVER
	solver = std::make_shared<ClothSolverGPU>();
#else
	solver = std::make_shared<ClothSolverCPU>();
#endif // USE_GPU_SOLVER

	//std::shared_ptr<ClothSolverBase> solver = std::make_shared<ClothSolverCPU>();
	//std::shared_ptr<ClothSolverBase> solver = std::make_shared<ClothSolverGPU>();
	Collider* sphereCollider = dynamic_cast<Collider*>(sphere.get());
	if (sphereCollider)
		solver->m_Colliders.push_back(sphereCollider);
	else
	{
		assert(false);
	}
	cloth->AddSolver(solver.get());
	m_Solvers.push_back(solver);
}
