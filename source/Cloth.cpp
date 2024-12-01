#include "Cloth.h"
#include <GL/glut.h>

Cloth::Cloth(float width, float height, int num_width, int num_height, bool fixed) : m_NumWidth(num_width), m_NumHeight(num_height), m_Width(width), m_Height(height), m_Fixed(fixed)
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
	glPushMatrix();

	glBegin(GL_TRIANGLES);
	for (int w = 0; w < m_NumWidth - 1; w++) {
		for (int h = 0; h < m_NumHeight - 1; h++) {
			glm::vec3 col(1.0f, 0.6f, 0.6f);
			if (w % 2 == h % 2) { col = glm::vec3(1.0f, 1.0f, 1.0f); }
			DrawTriangle(GetParticle(w + 1, h), GetParticle(w, h), GetParticle(w, h + 1), col);
			DrawTriangle(GetParticle(w + 1, h + 1), GetParticle(w + 1, h), GetParticle(w, h + 1), col);
		}
	}
	glEnd();

	glPopMatrix();
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
	glColor3fv((GLfloat*)&color);
	glVertex3fv((GLfloat*)&(p1->GetPosition()));
	glVertex3fv((GLfloat*)&(p2->GetPosition()));
	glVertex3fv((GLfloat*)&(p3->GetPosition()));
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
