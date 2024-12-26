#version 330 core

layout (location = 0) in vec3 aPos;  // matches glVertexAttribPointer(...) location
uniform mat4 uMVP;                   // a 4x4 MVP matrix uniform

void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
}