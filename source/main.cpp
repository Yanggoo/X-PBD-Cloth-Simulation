#include <memory>
#include "Application.h" 

void Display() {

}
int main(int argc, char* argv[])
{
	std::unique_ptr<Application> app = std::make_unique<Application>();
	app->Initialize(argc, argv);
	app->Run();
	app->End();
	return 0;
}

