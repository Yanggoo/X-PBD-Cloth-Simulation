#include "Scene.h"

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <filesystem>

#include "Cloth.h"
#include "Sphere.h"
#include "Globals.h"
#include "Cube.h"
#include "Application.h"

unsigned int Scene::compileShader(const char* filePath, GLenum type) {
	//std::filesystem::path currentPath = std::filesystem::current_path();
	//std::cout << "Current working directory: " << currentPath << std::endl;
	//return 0;

	std::string source;
	std::ifstream fileStream(filePath);
	if (!fileStream.is_open()) {
		std::cerr << "Could not open the file: " << filePath << std::endl;
		return 0;
	}

	std::stringstream buffer;
	buffer << fileStream.rdbuf();
	source = buffer.str();
    const char* source_cstr = source.c_str();
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source_cstr, nullptr);
    glCompileShader(shader);
    // Check for compile errors...
	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success); // Query compile status
	if (!success) {
		// Compilation failed, retrieve log
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

    return shader;
}

Scene::Scene(Application* application)
    : m_application(application)
{
	// Vertex shader
	unsigned int vertexShader = compileShader("source/shaders/vertex.glsl", GL_VERTEX_SHADER);

	// Fragment shader
	unsigned int fragmentShader = compileShader("source/shaders/fragment.glsl", GL_FRAGMENT_SHADER);

	// Link into shader program
	m_shaderProgram = glCreateProgram();
	glAttachShader(m_shaderProgram, vertexShader);
	glAttachShader(m_shaderProgram, fragmentShader);
	glLinkProgram(m_shaderProgram);
	
	// Check link status
	GLint success;
	glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetProgramInfoLog(m_shaderProgram, 512, nullptr, infoLog);
		std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}

	// Once linked, we can delete the intermediate shader objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	{
		// Vertex shader
		unsigned int vertexShader = compileShader("source/shaders/scene_vertex.glsl", GL_VERTEX_SHADER);

		// Fragment shader
		unsigned int fragmentShader = compileShader("source/shaders/scene_fragment.glsl", GL_FRAGMENT_SHADER);

		// Link into shader program
		m_sceneShaderProgram = glCreateProgram();
		glAttachShader(m_sceneShaderProgram, vertexShader);
		glAttachShader(m_sceneShaderProgram, fragmentShader);
		glLinkProgram(m_sceneShaderProgram);

		// Check link status
		GLint success;
		glGetProgramiv(m_sceneShaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			char infoLog[512];
			glGetProgramInfoLog(m_sceneShaderProgram, 512, nullptr, infoLog);
			std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}

		// Once linked, we can delete the intermediate shader objects
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}
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
	//glUseProgram(m_shaderProgram);

	//int mvpLocation = glGetUniformLocation(m_shaderProgram, "uMVP");
	//glm::mat4 MVP = projection * view * model;
	//glUseProgram(shaderProgram);
	//glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, glm::value_ptr(m_application->m_mvp));

	for (auto actor : m_Actors)
	{
		actor->Draw();
	}
}

void Scene::LoadSceneSphereAndCloth()
{
	m_Actors.clear();
	m_Solvers.clear();
	std::shared_ptr<Sphere> sphere = std::make_shared<Sphere>(this, 0.1f);
	sphere->Initialize(glm::vec3(0, 0, 1), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(sphere);
	std::shared_ptr<Cloth> cloth;

	std::shared_ptr<ClothSolverBase> solver;
#if USE_GPU_SOLVER
	solver = std::make_shared<ClothSolverGPU>();
	cloth = std::make_shared<Cloth>(this, 2, 2, 64, 64, true, glm::vec3(1.0f, 0.6f, 0.6f));
	cloth->Initialize(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(cloth);
#else
	solver = std::make_shared<ClothSolverCPU>();
	cloth = std::make_shared<Cloth>(this, 2, 2, 16, 16, true, glm::vec3(1.0f, 0.6f, 0.6f));
	cloth->Initialize(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(cloth);
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
	std::shared_ptr<Cube> platform = std::make_shared<Cube>(this, 3, 0.2, 2);
	platform->Initialize(glm::vec3(0, -1, 0), glm::vec3(1, 1, 1), glm::vec3(0, 0, 0));
	AddActor(platform);
	std::shared_ptr<Cloth> cloth0;

	std::shared_ptr<Cloth> cloth1;

	std::shared_ptr<ClothSolverBase> solver;
#if USE_GPU_SOLVER
	solver = std::make_shared<ClothSolverGPU>();
	cloth0 = std::make_shared<Cloth>(this, 1, 1, 32, 32, false, glm::vec3(1.0f, 0.6f, 0.6f));
	cloth0->Initialize(glm::vec3(-1, 0, 0), glm::vec3(1, 1, 1), glm::vec3(90, 0, 0));
	AddActor(cloth0);
	cloth1 = std::make_shared<Cloth>(this, 1, 1, 32, 32, true, glm::vec3(0.6f, 1.0f, 0.6f));
	cloth1->Initialize(glm::vec3(1, 0, 0), glm::vec3(1, 1, 1), glm::vec3(45, 90, 0));
	AddActor(cloth1);
#else
	cloth0 = std::make_shared<Cloth>(this, 1, 1, 16, 16, true, glm::vec3(1.0f, 0.6f, 0.6f));
	cloth0->Initialize(glm::vec3(-1, 0, 0), glm::vec3(1, 1, 1), glm::vec3(90, 0, 0));
	AddActor(cloth0);
	cloth1 = std::make_shared<Cloth>(this, 1, 1, 16, 16, true, glm::vec3(0.6f, 1.0f, 0.6f));
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

glm::mat4 Scene::GetMVP(float x, float y, float z) {
	float width = m_application->GetWidth();
    float height = m_application->GetHeight();
	float aspectRatio = (float)width / (float)height;
	glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
	glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::mat4 projection = glm::perspective(glm::radians(30.0f), aspectRatio, 0.001f, 1000.0f);
	glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
	glm::mat4 mvp = projection * view * model;
    return mvp;
}
