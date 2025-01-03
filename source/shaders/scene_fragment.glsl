#version 330 core

in vec4 FragPos; // The position of the fragment
in vec4 FragNormal;  // The normal of the fragment

out vec4 FragColor;

void main()
{
    vec4 lightDir = normalize(vec4(1.0, 1.0, 1.0, 0.0));
    vec4 baseColor = vec4(0.0, 0.0, 1.0, 1.0);
    float ambientStrength = 0.1;
    float diffuseStrength = 0.5;
    float specularStrength = 0.5;

    float NdotL = max(dot(FragNormal, lightDir), 0.0);
    vec4 ambient = ambientStrength * baseColor;
    vec4 diffuse = diffuseStrength * baseColor * NdotL;
    vec4 reflectDir = reflect(-lightDir, FragNormal);
    vec4 viewDir = normalize(vec4(0.0, 0.0, 0.0, 1.0) - FragPos);
    vec4 halfDir = normalize(lightDir + viewDir);
    float NdotH = max(dot(FragNormal, halfDir), 0.0);
    vec4 specular = specularStrength * baseColor * pow(NdotH, 32.0);

    vec4 color = ambient + diffuse + specular;
    FragColor = color;
}