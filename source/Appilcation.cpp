#include "Application.h"
#include <GL/freeglut.h>
#include "Timer.h"

// Forward declaration of the static display function
void DisplayWrapper();
void ReshapeWrapper(int width, int height);
void KeyBoardWrapper(unsigned char key, int x, int y);
void SpecialWrapper(int key, int x, int y);
void MouseWrapper(int button, int state, int x, int y);
void MouseMotionWrapper(int x, int y);

Application* appInstance = nullptr;

void Application::Initialize(int argc, char* argv[])
{
	//Config OpenGL
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(640, 480);
	glutCreateWindow("Cloth Simulation");
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glEnable(GL_CULL_FACE);
	
	// Set the global instance pointer to this instance
	appInstance = this;
	
	// Register the event callback function
	glutDisplayFunc(DisplayWrapper);
	glutReshapeFunc(ReshapeWrapper);
	glutKeyboardFunc(KeyBoardWrapper);
	glutSpecialFunc(SpecialWrapper);
	glutMouseFunc(MouseWrapper);
	glutMotionFunc(MouseMotionWrapper);

	// Initialize the scene
	m_Scene.LoadSceneSphereAndCloth();
	//m_Scene.LoadSceneClothAndCloth();
}

void Application::Run()
{
	m_Timer.Reset();
	m_AppRunning = true;
	while (m_AppRunning)
	{
		Idle();
		glutMainLoopEvent();
	}
}

void Application::End()
{

}

void Application::Reshape(int width, int height)
{
	static GLfloat lightPosition[4] = { 0.0f,  2.5f,  5.5f, 1.0f };
	static GLfloat lightDiffuse[3] = { 1.0f,  1.0f,  1.0f };
	static GLfloat lightAmbient[3] = { 0.25f, 0.25f, 0.25f };
	static GLfloat lightSpecular[3] = { 1.0f,  1.0f,  1.0f };

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glShadeModel(GL_SMOOTH);

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30.0, (double)width / (double)height, 0.0001f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(0.0f, 0.0f, 5.0f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); // pos, tgt, up

	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
	glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
	glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);
}

void Application::Idle(void)
{
	m_Timer.Update();
	m_Scene.Update(m_Timer.GetDeltaTime());
	if (m_Timer.NeedsFixedUpdate())
	{
		m_Scene.FixedUpdate(m_Timer.GetFixedUpdateInterval());
		glutPostRedisplay();
	}
}

void Application::KeyBoard(unsigned char key, int x, int y)
{
	// Implementation of KeyBoard
	if (key == 27)
	{
		m_AppRunning = false;
	}
}

void Application::Special(int key, int x, int y)
{
	// Implementation of Special
}

void Application::Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glDepthFunc(GL_LESS);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_NORMALIZE);

	m_Scene.Draw();

	glutSwapBuffers();
}

void Application::Mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		isMouseDown = true;
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		isMouseDown = false;
		if (m_SelectedParticle)
		{
			m_SelectedParticle->m_InvMass = 1.0f;
			m_SelectedParticle = nullptr;
		}
	}
}

void Application::MouseMotion(int x, int y)
{
	if (isMouseDown)
	{
		if (m_SelectedParticle)
		{
			m_SelectedParticle->SetPosition(m_Scene.getSolvers().front().get()->MouseToWorldZ0(x, y, selectedParticleZ));
			m_Scene.getSolvers().front().get()->setSelectedParticlePosition(m_SelectedParticle);
		}
		else
		{
			m_SelectedParticle = m_Scene.getSolvers().front().get()->GetParticleAtScreenPos(x, y);
			if (m_SelectedParticle != nullptr)
			{
				selectedParticleZ = m_SelectedParticle->m_Position.z;
				m_SelectedParticle->m_InvMass = 0.0f;
			}
		}
	}
}

// Static display function that calls the instance's display method
void DisplayWrapper()
{
	if (appInstance)
	{
		appInstance->Display();
	}
}

void ReshapeWrapper(int width, int height)
{
	if (appInstance)
	{
		appInstance->Reshape(width, height);
	}
}

void KeyBoardWrapper(unsigned char key, int x, int y)
{
	if (appInstance)
	{
		appInstance->KeyBoard(key, x, y);
	}
}

void SpecialWrapper(int key, int x, int y)
{
	if (appInstance)
	{
		appInstance->Special(key, x, y);
	}
}

void MouseWrapper(int button, int state, int x, int y)
{
	if (appInstance)
	{
		appInstance->Mouse(button, state, x, y);
	}
}

void MouseMotionWrapper(int x, int y)
{
	if (appInstance)
	{
		appInstance->MouseMotion(x, y);
	}
}
