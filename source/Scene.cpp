#include "Scene.h"
#include "Cloth.h"
#include "Sphere.h"
#include "Globals.h"
#include "Cube.h"

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

void Scene::LoadSceneSphereAndCloth()
{
	m_Actors.clear();
	m_Solvers.clear();
	std::shared_ptr<Sphere> sphere = std::make_shared<Sphere>(0.1f);
	sphere->Initialize(glm::vec3(0, 0, 1), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(sphere);
	std::shared_ptr<Cloth> cloth = std::make_shared<Cloth>(2, 2, 32, 32,true,glm::vec3(1.0f, 0.6f, 0.6f));
	cloth->Initialize(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(cloth);

	std::shared_ptr<ClothSolverBase> solver;
#if USE_GPU_SOLVER
	solver = std::make_shared<ClothSolverGPU>();
#else
	solver = std::make_shared<ClothSolverCPU>();
	dynamic_cast<ClothSolverCPU*>(solver.get())->m_MinDistanceBetweenParticles = 0.05;
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
	solver->OnInitFinish();
	m_Solvers.push_back(solver);
}

void Scene::LoadSceneClothAndCloth()
{
	m_Actors.clear();
	m_Solvers.clear();
	std::shared_ptr<Cube> platform = std::make_shared<Cube>(3, 0.2, 2);
	platform->Initialize(glm::vec3(0, -1, 0), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(platform);
	std::shared_ptr<Cloth> cloth0;

	std::shared_ptr<Cloth> cloth1;

	std::shared_ptr<ClothSolverBase> solver;
#if USE_GPU_SOLVER
	solver = std::make_shared<ClothSolverGPU>();
	cloth0 = std::make_shared<Cloth>(1, 1, 32, 32, true, glm::vec3(1.0f, 0.6f, 0.6f));
	cloth0->Initialize(glm::vec3(-1, 0, 0), glm::vec3(1, 1, 1), glm::vec3(90, 0, 0));
	AddActor(cloth0);
	cloth1 = std::make_shared<Cloth>(1, 1, 32, 32, true, glm::vec3(0.6f, 1.0f, 0.6f));
	cloth1->Initialize(glm::vec3(1, 0, 0), glm::vec3(1, 1, 1), glm::vec3(45, 90, 0));
	AddActor(cloth1);
	solver = std::make_shared<ClothSolverCPU>();
	dynamic_cast<ClothSolverCPU*>(solver.get())->m_MinDistanceBetweenParticles = 0.025;
#else
	cloth0 = std::make_shared<Cloth>(1, 1, 16, 16, true, glm::vec3(1.0f, 0.6f, 0.6f));
	cloth0->Initialize(glm::vec3(-1, 0, 0), glm::vec3(1, 1, 1), glm::vec3(90, 0, 0));
	AddActor(cloth0);
	cloth1 = std::make_shared<Cloth>(1, 1, 16, 16, true, glm::vec3(0.6f, 1.0f, 0.6f));
	cloth1->Initialize(glm::vec3(1, 0, 0), glm::vec3(1, 1, 1), glm::vec3(45, 90, 0));
	AddActor(cloth1);
	solver = std::make_shared<ClothSolverCPU>();
	dynamic_cast<ClothSolverCPU*>(solver.get())->m_MinDistanceBetweenParticles = 0.05;
#endif // USE_GPU_SOLVER

	//std::shared_ptr<ClothSolverBase> solver = std::make_shared<ClothSolverCPU>();
	//std::shared_ptr<ClothSolverBase> solver = std::make_shared<ClothSolverGPU>();
	Collider* boxCollider = dynamic_cast<Collider*>(platform.get());
	if (boxCollider)
		solver->m_Colliders.push_back(boxCollider);
	else
	{
		assert(false);
	}
	cloth0->AddSolver(solver.get());
	cloth1->AddSolver(solver.get());
	solver->OnInitFinish();
	m_Solvers.push_back(solver);
}
