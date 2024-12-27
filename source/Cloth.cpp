#include "Cloth.h"

#include <GL/glew.h>
#include <GL/glut.h>

#include "Scene.h"

Cloth::Cloth(Scene* scene, float width, float height, int num_width, int num_height, bool fixed, glm::vec3 color) : m_NumWidth(num_width), m_NumHeight(num_height), 
m_Width(width), m_Height(height), m_Fixed(fixed), m_Color(color), m_Scene(scene)
{
	assert(m_NumWidth > 1 && m_NumHeight > 1);
	m_Particles.reserve(m_NumWidth * m_NumHeight);
	for (int w = 0; w < m_NumWidth; w++) {
		for (int h = 0; h < m_NumHeight; h++) {
			glm::vec3 pos = glm::vec3(w * (float)m_Width / (m_NumWidth - 1) - 0.5f * m_Width, -h * (float)m_Height / (m_NumHeight-1) + 0.5f * m_Height, 0.0f);
			float inv_mass = 1.0f;
			if (m_Fixed &&((h == 0) && (w == 0) ||
				(h == 0) && (w == m_NumWidth - 1))) {
				inv_mass = 0.0f; //fix only edge point
			}
			m_Particles.emplace_back(inv_mass, pos);
		}
	}

	int indexCount = 6 * (m_NumWidth - 1) * (m_NumHeight - 1);
    unsigned int* indices = new unsigned int[indexCount];
	for (int w = 0; w < m_NumWidth - 1; w++) {
		for (int h = 0; h < m_NumHeight - 1; h++) {
            indices[(w * (m_NumHeight - 1) + h) * 6 + 0] = w * m_NumHeight + h;
            indices[(w * (m_NumHeight - 1) + h) * 6 + 1] = w * m_NumHeight + h + 1;
            indices[(w * (m_NumHeight - 1) + h) * 6 + 2] = (w + 1) * m_NumHeight + h;

            indices[(w * (m_NumHeight - 1) + h) * 6 + 3] = w * m_NumHeight + h + 1;
            indices[(w * (m_NumHeight - 1) + h) * 6 + 4] = (w + 1) * m_NumHeight + h + 1;
            indices[(w * (m_NumHeight - 1) + h) * 6 + 5] = (w + 1) * m_NumHeight + h;
		}
	}

	//unsigned int VAO;
	//unsigned int VBO;
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_vbo);

	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

	glBufferData(GL_ARRAY_BUFFER, m_Particles.size() * sizeof(Particle), m_Particles.data(), GL_STATIC_DRAW);

	unsigned int EBO;
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), indices, GL_STATIC_DRAW);

	// location 0 vec3 position
	glEnableVertexAttribArray(0); // enable the attribute at location 0
	glVertexAttribPointer(
		0,               // location of the attribute in the vertex shader
		3,               // number of components (x, y, z)
		GL_FLOAT,        // data type
		GL_FALSE,        // whether to normalize
		sizeof(Particle),// stride (distance between consecutive vertex attributes)
		(void*) 0        // offset in the array (where does the position data start?)
	);

	glEnableVertexAttribArray(1); // enable the attribute at location 0
	glVertexAttribPointer(
		1,               // location of the attribute in the vertex shader
		3,               // number of components (x, y, z)
		GL_FLOAT,        // data type
		GL_FALSE,        // whether to normalize
		sizeof(Particle),// stride (distance between consecutive vertex attributes)
		(void*)offsetof(Particle, m_Normal)        // offset in the array (where does the position data start?)
	);

	glBindVertexArray(0);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Cloth::~Cloth()
{

}

void Cloth::Update(float deltaTime)
{

}

void Cloth::FixedUpdate(float deltaTime)
{
}

void Cloth::Draw()
{
    glUseProgram(m_Scene->GetShaderProgram());
	glBindVertexArray(m_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

	UpdateNormals();
	glBufferData(GL_ARRAY_BUFFER, m_Particles.size() * sizeof(Particle), &m_Particles[0], GL_STATIC_DRAW);

	auto mvp = m_Scene->GetMVP(0.f, 0.f, 0.f);
	int mvpLocation = glGetUniformLocation(m_Scene->GetSceneShaderProgram(), "uMVP");
	glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));

	// 5. Issue the draw call
	glDrawElements(GL_TRIANGLES, (m_NumWidth - 1) * (m_NumHeight - 1) * 6, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);

	//glPushMatrix();

	//glBegin(GL_TRIANGLES);
	//for (int w = 0; w < m_NumWidth - 1; w++) {
	//	for (int h = 0; h < m_NumHeight - 1; h++) {
	//		glm::vec3 col = m_Color;
	//		int wIdx = w * 6 / m_NumWidth;
	//		int hIdx = h * 6 / m_NumHeight;
	//		if (wIdx % 2 == hIdx % 2)
	//		{ col = glm::vec3(1.0f, 1.0f, 1.0f); }
	//		DrawTriangle(GetParticle(w + 1, h), GetParticle(w, h), GetParticle(w, h + 1), col);
	//		DrawTriangle(GetParticle(w + 1, h + 1), GetParticle(w + 1, h), GetParticle(w, h + 1), col);
	//	}
	//}
	//glEnd();

	//glPopMatrix();
}

void Cloth::Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation)
{
	Actor::Initialize(position, scale, rotation);
	glm::mat4 model = m_Transform.matrix();
	for (auto& particle : m_Particles) {
		auto pos = particle.GetPosition();
		particle.SetPosition(glm::vec3(model * glm::vec4(pos, 1.0f)));
	}
}

void Cloth::DrawTriangle(Particle* p1, Particle* p2, Particle* p3, const glm::vec3 color)
{
	//glColor3fv((GLfloat*)&color);
	//glVertex3fv((GLfloat*)&(p1->GetPosition()));
	//glVertex3fv((GLfloat*)&(p2->GetPosition()));
	//glVertex3fv((GLfloat*)&(p3->GetPosition()));
}

Particle* Cloth::GetParticle(int w, int h)
{
	return &m_Particles[w * m_NumWidth + h];
}

void Cloth::AddSolver(ClothSolverBase* solver)
{
	m_ClothSolver = solver;
	solver->ResponsibleFor(this);
}

void Cloth::UpdateNormals() {
	// Calculate face normals and accumulate
	for (size_t w = 0; w < m_NumWidth - 1; w++) {
		for (size_t h = 0; h < m_NumHeight - 1; h++) {
			glm::vec3& p0 = m_Particles[w * m_NumHeight + h].m_Position;
			glm::vec3& p1 = m_Particles[w * m_NumHeight + h + 1].m_Position;
			glm::vec3& p2 = m_Particles[(w + 1) * m_NumHeight + h].m_Position;
			glm::vec3& p3 = m_Particles[(w + 1) * m_NumHeight + h + 1].m_Position;

			glm::vec3 normal0 = glm::normalize(glm::cross(p1 - p0, p2 - p0));
			glm::vec3 normal1 = glm::normalize(glm::cross(p2 - p3, p1 - p3));

			m_Particles[w * m_NumHeight + h].m_Normal += normal0;
			m_Particles[w * m_NumHeight + h + 1].m_Normal += normal0 + normal1;
			m_Particles[(w + 1) * m_NumHeight + h].m_Normal += normal0 + normal1;
			m_Particles[(w + 1) * m_NumHeight + h + 1].m_Normal += normal1;
		}
	}

	// Normalize the accumulated normals
	for (auto& particle : m_Particles) {
		particle.m_Normal = glm::normalize(particle.m_Normal);
	}
}
