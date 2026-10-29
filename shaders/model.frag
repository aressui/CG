#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

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

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
};

struct SpotLight {
    vec3 position;
    float intensity;
    vec3 direction;
    float cosInner;
    vec3 color;
    float cosOuter;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight pointLights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spotLights[];
};

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec3 albedo;
    float shininess;
    vec3 specular;
    float useVertexColor;
} obj;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = diff > 0.0 ? pow(max(dot(N, H), 0.0), obj.shininess) : 0.0;
    return radiance * (diff * fragColor + spec * obj.specular);
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos - fragPosition);

    vec3 color = scene.ambientColor * scene.ambientIntensity * fragColor;
    color += blinnPhong(N, V, -scene.dirDirection, scene.dirColor * scene.dirIntensity);

    for (uint i = 0; i < scene.pointCount; ++i) {
        PointLight p = pointLights[i];
        vec3 toLight = p.position - fragPosition;
        float dist2 = max(dot(toLight, toLight), 1e-4);
        vec3 L = toLight / sqrt(dist2);
        color += blinnPhong(N, V, L, p.color * p.intensity / dist2);
    }

    for (uint i = 0; i < scene.spotCount; ++i) {
        SpotLight s = spotLights[i];
        vec3 toLight = s.position - fragPosition;
        float dist2 = max(dot(toLight, toLight), 1e-4);
        vec3 L = toLight / sqrt(dist2);
        float cosAngle = dot(-L, s.direction);
        float cone = s.cosInner > s.cosOuter
            ? clamp((cosAngle - s.cosOuter) / (s.cosInner - s.cosOuter), 0.0, 1.0)
            : step(s.cosOuter, cosAngle);
        color += blinnPhong(N, V, L, s.color * s.intensity * cone / dist2);
    }

    outColor = vec4(color, 1.0);
}
