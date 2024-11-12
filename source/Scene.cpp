#include "Scene.h"
#include "Cloth.h"
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
	std::shared_ptr<Cloth> cloth = std::make_shared<Cloth>(2, 2, 20, 20);
	cloth->Initialize(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(cloth);
}
