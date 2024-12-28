#include "Cloth.h"

#include <GL/glew.h>
#include <GL/glut.h>

#include "Scene.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

            glm::vec2 uv = glm::vec2(w / (float)(m_NumWidth - 1), h / (float)(m_NumHeight - 1));
			m_Particles.emplace_back(inv_mass, pos, glm::vec3(0, 0, 0), uv);
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

	glEnableVertexAttribArray(2); // enable the attribute at location 0
	glVertexAttribPointer(
		2,               // location of the attribute in the vertex shader
		2,               // number of components (x, y, z)
		GL_FLOAT,        // data type
		GL_FALSE,        // whether to normalize
		sizeof(Particle),// stride (distance between consecutive vertex attributes)
		(void*)offsetof(Particle, m_TexCoord)        // offset in the array (where does the position data start?)
	);

	glEnableVertexAttribArray(3); // enable the attribute at location 0
	glVertexAttribPointer(
		3,               // location of the attribute in the vertex shader
		4,               // number of components (x, y, z)
		GL_FLOAT,        // data type
		GL_FALSE,        // whether to normalize
		sizeof(Particle),// stride (distance between consecutive vertex attributes)
		(void*)offsetof(Particle, m_Tangent)        // offset in the array (where does the position data start?)
	);

	glBindVertexArray(0);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	{
		int width, height, channels;
		unsigned char* data = stbi_load("source/assets/base.jpg", &width, &height, &channels, 0);
		if (!data) {
			std::cerr << "Failed to load texture" << std::endl;
			// handle error
		}

		glGenTextures(1, &m_texId);
		glBindTexture(GL_TEXTURE_2D, m_texId);
		

		// upload the image data
		// if channels=3, format = GL_RGB; if channels=4, format = GL_RGBA
		GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

		// generate mipmaps
		glGenerateMipmap(GL_TEXTURE_2D);

		// set parameters (wrapping, filtering)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		// done with data on CPU
		stbi_image_free(data);

		// unbind
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	
	{
		int width, height, channels;
		unsigned char* data = stbi_load("source/assets/normal.jpg", &width, &height, &channels, 0);
		if (!data) {
			std::cerr << "Failed to load texture" << std::endl;
			// handle error
		}

		glGenTextures(1, &m_texNormalId);
		glBindTexture(GL_TEXTURE_2D, m_texNormalId);


		// upload the image data
		// if channels=3, format = GL_RGB; if channels=4, format = GL_RGBA
		GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

		// generate mipmaps
		glGenerateMipmap(GL_TEXTURE_2D);

		// set parameters (wrapping, filtering)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		// done with data on CPU
		stbi_image_free(data);

		// unbind
		glBindTexture(GL_TEXTURE_2D, 0);
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
	auto shaderProgram = m_Scene->GetShaderProgram();
    glUseProgram(shaderProgram);
	glBindVertexArray(m_vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

	UpdateNormals();
	UpdateTangents();
	glBufferData(GL_ARRAY_BUFFER, m_Particles.size() * sizeof(Particle), &m_Particles[0], GL_STATIC_DRAW);

	auto mvp = m_Scene->GetMVP(0.f, 0.f, 0.f);
	int mvpLocation = glGetUniformLocation(shaderProgram, "uMVP");
	glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));

	GLint loc = glGetUniformLocation(shaderProgram, "uTexBaseColor");
	glUniform1i(loc, 0); // means "texture unit 0"

	// Activate texture unit 0, bind the texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texId);

	{
		GLint loc = glGetUniformLocation(shaderProgram, "uTexNormal");
		glUniform1i(loc, 1); // means "texture unit 0"

		// Activate texture unit 0, bind the texture
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, m_texNormalId);
	}

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

void Cloth::UpdateTangents() {
	for (size_t w = 0; w < m_NumWidth - 1; w++) {
		for (size_t h = 0; h < m_NumHeight - 1; h++) {
            glm::vec3& p0 = m_Particles[w * m_NumHeight + h].m_Position;
            glm::vec3& p1 = m_Particles[w * m_NumHeight + h + 1].m_Position;
            glm::vec3& p2 = m_Particles[(w + 1) * m_NumHeight + h].m_Position;
            glm::vec2& uv0 = m_Particles[w * m_NumHeight + h].m_TexCoord;
            glm::vec2& uv1 = m_Particles[w * m_NumHeight + h + 1].m_TexCoord;
            glm::vec2& uv2 = m_Particles[(w + 1) * m_NumHeight + h].m_TexCoord;
            glm::vec3 dp1 = p1 - p0;
            glm::vec3 dp2 = p2 - p0;
            glm::vec2 duv1 = uv1 - uv0;
            glm::vec2 duv2 = uv2 - uv0;
			float r = 1.0f / (duv1.x * duv2.y - duv2.x * duv1.y);
            glm::vec3 tangent = (dp1 * duv2.y - dp2 * duv1.y) * r;
            glm::vec3 bitangent = (dp2 * duv1.x - dp1 * duv2.x) * r;
            glm::vec3 normal = m_Particles[w * m_NumHeight + h].m_Normal;
            tangent = tangent - normal * glm::dot(normal, tangent);
			float handedness = (glm::dot(glm::cross(normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
            glm::vec4 combinedTangent = glm::normalize(glm::vec4(tangent, handedness));
            m_Particles[w * m_NumHeight + h].m_Tangent = combinedTangent;

			if (w == m_NumWidth - 2) {
                m_Particles[(w + 1) * m_NumHeight + h].m_Tangent = combinedTangent;
			}

            if (h == m_NumHeight - 2) {
                m_Particles[w * m_NumHeight + h + 1].m_Tangent = combinedTangent;
            }

            if (w == m_NumWidth - 2 && h == m_NumHeight - 2) {
                m_Particles[(w + 1) * m_NumHeight + h + 1].m_Tangent = combinedTangent;
            }
		}
	}
}
