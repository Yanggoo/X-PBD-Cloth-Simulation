#version 330 core

layout (location = 0) in vec3 aPos;  // matches glVertexAttribPointer(...) location
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;
layout (location = 3) in vec4 aTangent;

out vec4 FragPos;
out vec4 FragNormal;
out vec2 FragUV;
out vec3 FragTangent;
out vec3 FragBitangent;

uniform mat4 uMVP;                   // a 4x4 MVP matrix uniform

void main()
{
    vec4 position = uMVP * vec4(aPos, 1.0);
    FragPos = position;
    FragNormal = vec4(aNormal, 0.0);
    FragUV = aUV;
    FragTangent = aTangent.xyz;
    FragBitangent = normalize(cross(aNormal, aTangent.xyz) * aTangent.w);
    gl_Position = position;
}