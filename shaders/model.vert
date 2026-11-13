#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    vec3 cameraPos;
    uint pointCount;
    vec3 ambientColor;
    float ambientIntensity;
    vec3 dirDirection;
    uint spotCount;
    vec3 dirColor;
    float dirIntensity;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec3 specular;
    float shininess;
    vec2 uvScale;
} obj;

void main() {
    vec4 worldPos = obj.model * vec4(inPosition, 1.0);
    gl_Position = scene.viewProj * worldPos;
    fragPosition = worldPos.xyz;
    fragNormal = transpose(inverse(mat3(obj.model))) * inNormal;
    fragUV = inUV * obj.uvScale;
}
