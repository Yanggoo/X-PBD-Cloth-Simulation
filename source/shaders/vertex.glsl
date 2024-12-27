#version 330 core

layout (location = 0) in vec3 aPos;  // matches glVertexAttribPointer(...) location
layout (location = 1) in vec3 aNormal;

out vec4 FragPos;
out vec4 FragNormal;

uniform mat4 uMVP;                   // a 4x4 MVP matrix uniform

void main()
{
    vec4 position = uMVP * vec4(aPos, 1.0);
    FragPos = position;
    FragNormal = vec4(aNormal, 0.0);
    gl_Position = position;
}