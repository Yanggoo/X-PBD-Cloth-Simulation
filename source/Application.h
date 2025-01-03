#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Timer.h"
#include "Scene.h"
class Application
{
public:
	Application() :m_Timer(), m_IterationNum(20), m_Window(-1), m_AppRunning(false), m_Scene(this){};
	void Initialize(int argc, char* argv[]);
	void Run();
	void End();
	void Reshape(int width, int height);
	void Idle(void);
	void KeyBoard(unsigned char key, int x, int y);
	void Special(int key, int x, int y);
	void Display(void);
	void Mouse(int button, int state, int x, int y);
	void MouseMotion(int x, int y);

    float GetWidth() { return m_width; }
    float GetHeight() { return m_height; }

private:
	ClothSimulation::Timer	m_Timer;
	int		m_IterationNum;
	int		m_Window;
	bool	m_AppRunning;
	Scene	m_Scene;
	Particle* m_SelectedParticle;
	glm::vec3 m_SelectedParticleDestinPosition;
	bool isMouseDown = false;
	float selectedParticleZ;
	float m_width;
	float m_height;

//public:
//	glm::mat4 m_mvp;
};

