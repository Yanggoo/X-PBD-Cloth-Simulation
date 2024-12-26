#include <memory>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Application.h" 

void Display() {

}
int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitContextVersion(3, 3);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(640, 480);
	glutCreateWindow("Cloth Simulation");

	glewExperimental = GL_TRUE; // recommended for core profile
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
		return -1;
	}

	std::unique_ptr<Application> app = std::make_unique<Application>();
	app->Initialize(argc, argv);
	app->Run();
	app->End();
	return 0;
}

