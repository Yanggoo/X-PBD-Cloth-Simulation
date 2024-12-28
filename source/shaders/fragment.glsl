#version 330 core

in vec4 FragPos; // The position of the fragment
in vec4 FragNormal;  // The normal of the fragment
in vec2 FragUV;
in vec3 FragTangent;
in vec3 FragBitangent;

out vec4 FragColor;

uniform sampler2D uTexBaseColor;
uniform sampler2D uTexNormal;

void main()
{
    vec4 faceNormal = normalize(gl_FrontFacing ? FragNormal : -FragNormal);
    mat3 TBN = mat3(FragTangent, FragBitangent, faceNormal);
    vec3 normalMap = texture(uTexNormal, FragUV).rgb;
    normalMap = normalize(normalMap * 2.0 - 1.0);

    vec4 normal = vec4(normalize(TBN * normalMap), 0.0);
    // Reverse the normal for back faces
    // vec4 normal = gl_FrontFacing ? normalize(FragNormal) : -normalize(FragNormal);

    vec4 lightDir = normalize(vec4(1.0, 1.0, 1.0, 0.0));
    // vec4 baseColor = vec4(1.0, 0.6, 0.6, 1.0);
    vec4 baseColor = texture(uTexBaseColor, FragUV);
    float ambientStrength = 0.1;
    float diffuseStrength = 0.5;
    float specularStrength = 0.5;
    float sheenIntensity = 0.5;

    float NdotL = max(dot(normal, lightDir), 0.0);
    vec4 viewDir = normalize(vec4(0.0, 0.0, 0.0, 1.0) - FragPos);
    vec4 halfDir = normalize(lightDir + viewDir);
    float NdotH = max(dot(normal, halfDir), 0.0);
    float NdotV = max(dot(normal, viewDir), 0.0);
    float fresnelTerm = pow(1.0 - NdotV, 5.0);
    fresnelTerm *= sheenIntensity;
    vec4 sheen = fresnelTerm * baseColor;

    vec4 ambient = ambientStrength * baseColor;
    vec4 diffuse = diffuseStrength * baseColor * NdotL;
    // vec4 reflectDir = reflect(-lightDir, normal);
    vec4 specular = specularStrength * baseColor * pow(NdotH, 32.0);

    vec4 color = ambient + diffuse + specular + sheen;
    FragColor = color;
}