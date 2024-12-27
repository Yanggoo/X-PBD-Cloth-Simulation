#include "Cube.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Scene.h"

Cube::Cube(Scene* scene, float width, float height, float depth)
    : m_Dimensions(width, height, depth), m_Position(glm::vec3(0.0f)), m_Scene(scene)
{
    CreateCubeData(width, height, depth, vertices, indices);

    GLuint EBO;
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &EBO);

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
}

Cube::~Cube() = default;

void Cube::Update(float deltaTime)
{

}

void Cube::FixedUpdate(float deltaTime)
{

}

// Draw method
void Cube::Draw()
{
    glUseProgram(m_Scene->GetSceneShaderProgram());
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    auto mvp = m_Scene->GetMVP(m_Position.x, m_Position.y, m_Position.z);
    int mvpLocation = glGetUniformLocation(m_Scene->GetSceneShaderProgram(), "uMVP");
    glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));


    //glBufferData(GL_ARRAY_BUFFER, m_Particles.size() * sizeof(Particle), &m_Particles[0], GL_STATIC_DRAW);

    // 5. Issue the draw call
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
}

void Cube::Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation)
{
	Actor::Initialize(position, scale, rotation);
	m_Position = position;
	m_Dimensions = glm::vec3(m_Dimensions.x * scale.x, m_Dimensions.y * scale.y, m_Dimensions.z * scale.z);
}

void Cube::CreateCubeData(float w, float h, float d,
    std::vector<float>& vertices,
    std::vector<unsigned int>& indices) {
    float halfW = w * 0.5f;
    float halfH = h * 0.5f;
    float halfD = d * 0.5f;

    // 24 vertices: position(x,y,z), normal(x,y,z), UV(u,v)
    float tempVertices[24 * 8] = {
        // FRONT (Z+)
        -halfW, -halfH,  halfD,   0.0f, 0.0f, +1.0f,   0.0f, 0.0f, // [0]
         halfW, -halfH,  halfD,   0.0f, 0.0f, +1.0f,   1.0f, 0.0f, // [1]
         halfW,  halfH,  halfD,   0.0f, 0.0f, +1.0f,   1.0f, 1.0f, // [2]
        -halfW,  halfH,  halfD,   0.0f, 0.0f, +1.0f,   0.0f, 1.0f, // [3]

        // BACK  (Z-)
        -halfW, -halfH, -halfD,   0.0f, 0.0f, -1.0f,   1.0f, 0.0f, // [4]
        -halfW,  halfH, -halfD,   0.0f, 0.0f, -1.0f,   1.0f, 1.0f, // [5]
         halfW,  halfH, -halfD,   0.0f, 0.0f, -1.0f,   0.0f, 1.0f, // [6]
         halfW, -halfH, -halfD,   0.0f, 0.0f, -1.0f,   0.0f, 0.0f, // [7]

         // LEFT  (X-)
         -halfW, -halfH, -halfD,  -1.0f, 0.0f, 0.0f,    0.0f, 0.0f, // [8]
         -halfW, -halfH,  halfD,  -1.0f, 0.0f, 0.0f,    1.0f, 0.0f, // [9]
         -halfW,  halfH,  halfD,  -1.0f, 0.0f, 0.0f,    1.0f, 1.0f, // [10]
         -halfW,  halfH, -halfD,  -1.0f, 0.0f, 0.0f,    0.0f, 1.0f, // [11]

         // RIGHT (X+)
          halfW, -halfH, -halfD,  +1.0f, 0.0f, 0.0f,    1.0f, 0.0f, // [12]
          halfW,  halfH, -halfD,  +1.0f, 0.0f, 0.0f,    1.0f, 1.0f, // [13]
          halfW,  halfH,  halfD,  +1.0f, 0.0f, 0.0f,    0.0f, 1.0f, // [14]
          halfW, -halfH,  halfD,  +1.0f, 0.0f, 0.0f,    0.0f, 0.0f, // [15]

          // TOP   (Y+)
          -halfW,  halfH, -halfD,   0.0f, +1.0f, 0.0f,   0.0f, 0.0f, // [16]
          -halfW,  halfH,  halfD,   0.0f, +1.0f, 0.0f,   0.0f, 1.0f, // [17]
           halfW,  halfH,  halfD,   0.0f, +1.0f, 0.0f,   1.0f, 1.0f, // [18]
           halfW,  halfH, -halfD,   0.0f, +1.0f, 0.0f,   1.0f, 0.0f, // [19]

           // BOTTOM (Y-)
           -halfW, -halfH, -halfD,   0.0f, -1.0f, 0.0f,   1.0f, 1.0f, // [20]
            halfW, -halfH, -halfD,   0.0f, -1.0f, 0.0f,   0.0f, 1.0f, // [21]
            halfW, -halfH,  halfD,   0.0f, -1.0f, 0.0f,   0.0f, 0.0f, // [22]
           -halfW, -halfH,  halfD,   0.0f, -1.0f, 0.0f,   1.0f, 0.0f  // [23]
    };

    unsigned int tempIndices[36] = {
        // FRONT
         0,  1,  2,  2,  3,  0,
         // BACK
          4,  5,  6,  6,  7,  4,
          // LEFT
           8,  9, 10, 10, 11,  8,
           // RIGHT
           12, 13, 14, 14, 15, 12,
           // TOP
           16, 17, 18, 18, 19, 16,
           // BOTTOM
           20, 21, 22, 22, 23, 20
    };

    // Copy into vectors
    vertices.assign(tempVertices, tempVertices + 24 * 8);
    indices.assign(tempIndices, tempIndices + 36);
}

glm::vec3 Cube::ComputeSDF(glm::vec3 position)
{
	glm::vec3 halfExtents = m_Dimensions * 0.5f+0.1f;
	glm::vec3 diff = position - m_Position;
	if (diff.x > -halfExtents.x && diff.x<halfExtents.x
		&& diff.y>-halfExtents.y && diff.y < halfExtents.y
		&& diff.z>-halfExtents.z && diff.z < halfExtents.z) {
		float dx = diff.x > 0 ? halfExtents.x  - diff.x : -halfExtents.x  - diff.x;
		float dy = diff.y > 0 ? halfExtents.y  - diff.y : -halfExtents.y  - diff.y;
		float dz = diff.z > 0 ? halfExtents.z  - diff.z : -halfExtents.z  - diff.z;
		if (abs(dx) <= abs(dy) && abs(dx) <= abs(dz)) {
			return glm::vec3(dx, 0.0f, 0.0f);
		}
		else if (abs(dy) <= abs(dx) && abs(dy) <= abs(dz)) {
			return glm::vec3(0.0f, dy, 0.0f);
		}
		else {
			return glm::vec3(0.0f, 0.0f, dz);
		}
	}
	return glm::vec3(0.0f);

}

