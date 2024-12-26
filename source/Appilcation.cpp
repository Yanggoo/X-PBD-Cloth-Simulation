#include "Application.h"

#include "Timer.h"
#include "imgui.h"
//#include "imgui_impl_opengl2.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glut.h"

// Forward declaration of the static display functions
void DisplayWrapper();
void ReshapeWrapper(int width, int height);
void KeyBoardWrapper(unsigned char key, int x, int y);
void SpecialWrapper(int key, int x, int y);
void MouseWrapper(int button, int state, int x, int y);
void MouseMotionWrapper(int x, int y);

Application* appInstance = nullptr;

void Application::Initialize(int argc, char* argv[])
{
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// Set the global instance pointer to this instance
	appInstance = this;

	// Register event callback functions
	glutDisplayFunc(DisplayWrapper);
	glutReshapeFunc(ReshapeWrapper);
	glutKeyboardFunc(KeyBoardWrapper);
	glutSpecialFunc(SpecialWrapper);
	glutMouseFunc(MouseWrapper);
	glutMotionFunc(MouseMotionWrapper);

	// Initialize the scene
	m_Scene.LoadSceneSphereAndCloth();
	// m_Scene.LoadSceneClothAndCloth();

	// Initialize ImGui
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontDefault();
	ImGui_ImplGLUT_Init();
	ImGui_ImplGLUT_InstallFuncs();
	ImGui_ImplOpenGL3_Init();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

    Reshape(640, 480);

	// Re-register callback functions after ImGui installation
	glutDisplayFunc(DisplayWrapper);
	glutReshapeFunc(ReshapeWrapper);
	glutKeyboardFunc(KeyBoardWrapper);
	glutSpecialFunc(SpecialWrapper);
	glutMouseFunc(MouseWrapper);
	glutMotionFunc(MouseMotionWrapper);
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
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGLUT_Shutdown();
	ImGui::DestroyContext();
}

void Application::Reshape(int width, int height)
{
	//static GLfloat lightPosition[4] = { 0.0f, 2.5f, 5.5f, 1.0f };
	//static GLfloat lightDiffuse[3] = { 1.0f, 1.0f, 1.0f };
	//static GLfloat lightAmbient[3] = { 0.25f, 0.25f, 0.25f };
	//static GLfloat lightSpecular[3] = { 1.0f, 1.0f, 1.0f };

	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);

	//glShadeModel(GL_SMOOTH);

	glViewport(0, 0, width, height);

	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//gluPerspective(30.0, (double)width / (double)height, 0.0001f, 1000.0f);
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();

	//gluLookAt(0.0f, 0.0f, 5.0f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    float aspectRatio = (float)width / (float)height;
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::mat4 projection = glm::perspective(glm::radians(30.0f), aspectRatio, 0.001f, 1000.0f);
	glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
	glm::mat4 model = glm::mat4(1.0f);
    m_mvp = projection * view * model;

	//glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
	//glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
	//glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
	//glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);

	// Update ImGui DisplaySize
	ImGuiIO& io = ImGui::GetIO();
	io.DisplaySize = ImVec2((float)width, (float)height);
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

	//glEnable(GL_DEPTH_TEST);
	////glEnable(GL_LIGHTING);
	//glDepthFunc(GL_LESS);
	//glEnable(GL_COLOR_MATERIAL);
	//glEnable(GL_NORMALIZE);

	m_Scene.Draw();

	// ImGui rendering
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGLUT_NewFrame();
	ImGui::NewFrame();

	// Set GUI window position to the top-left corner
	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);

	// Optionally set window size
	ImGui::SetNextWindowSize(ImVec2(230, 80), ImGuiCond_Always);

	// Render ImGui window
	ImGui::Begin("Scene Selector", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

	if (ImGui::Button("Load Sphere and Cloth Scene"))
	{
		m_Scene.LoadSceneSphereAndCloth();
	}
	if (ImGui::Button("Load Cloth and Cloth Scene"))
	{
		m_Scene.LoadSceneClothAndCloth();
	}
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glutSwapBuffers();
}

void Application::Mouse(int button, int state, int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse)
	{
		return;
	}
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
			m_Scene.getSolvers().front()->OnInputClearParticle(m_SelectedParticle);
			m_SelectedParticle = nullptr;
		}
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		isMouseDown = false;
		if (m_SelectedParticle)
		{
			m_SelectedParticle->m_InvMass = 0.0f;
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

			m_Scene.getSolvers().front()->OnInputSelectParticle(m_SelectedParticle);
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
	ImGuiIO& io = ImGui::GetIO();

	io.MousePos = ImVec2((float)x, (float)y);

	if (button == GLUT_LEFT_BUTTON)
		io.MouseDown[0] = (state == GLUT_DOWN);
	else if (button == GLUT_RIGHT_BUTTON)
		io.MouseDown[1] = (state == GLUT_DOWN);

	if (!io.WantCaptureMouse && appInstance)
		appInstance->Mouse(button, state, x, y);
}

void MouseMotionWrapper(int x, int y)
{
	if (appInstance)
	{
		appInstance->MouseMotion(x, y);
	}
}
