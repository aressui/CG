#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

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
    vec3 specular;
    float shininess;
    vec2 uvScale;
} obj;

layout(set = 1, binding = 1) uniform sampler2D albedoTexture;

layout(push_constant) uniform SwirlEffect {
    uint enabled;
    float time;
    float strength;
    float radius;
} swirl;

const float PI = 3.14159265;

vec3 sampleAlbedo(vec2 uv) {
    if (swirl.enabled == 0u) {
        return texture(albedoTexture, uv).rgb;
    }

    vec2 tile = floor(uv);
    vec2 p = uv - tile - 0.5;
    float r = length(p);
    float falloff = max(1.0 - r / swirl.radius, 0.0);
    float angle = swirl.strength * falloff * falloff * sin(swirl.time);

    vec3 sum = vec3(0.0);
    for (int k = 0; k < 5; ++k) {
        float a = angle * (1.0 - 0.1 * float(k));
        vec2 q = vec2(cos(a) * p.x - sin(a) * p.y, sin(a) * p.x + cos(a) * p.y);
        sum += texture(albedoTexture, tile + 0.5 + q).rgb;
    }
    vec3 color = sum / 5.0;

    vec3 tint = 0.7 + 0.3 * cos(swirl.time + 2.0 * PI * r + vec3(0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0));
    return color * mix(vec3(1.0), tint, falloff);
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = diff > 0.0 ? pow(max(dot(N, H), 0.0), obj.shininess) : 0.0;
    return radiance * (diff * albedo + spec * obj.specular);
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos - fragPosition);
    vec3 albedo = sampleAlbedo(fragUV);

    vec3 color = scene.ambientColor * scene.ambientIntensity * albedo;
    color += blinnPhong(N, V, -scene.dirDirection, scene.dirColor * scene.dirIntensity, albedo);

    for (uint i = 0; i < scene.pointCount; ++i) {
        PointLight p = pointLights[i];
        vec3 toLight = p.position - fragPosition;
        float dist2 = max(dot(toLight, toLight), 1e-4);
        vec3 L = toLight / sqrt(dist2);
        color += blinnPhong(N, V, L, p.color * p.intensity / dist2, albedo);
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
        color += blinnPhong(N, V, L, s.color * s.intensity * cone / dist2, albedo);
    }

    outColor = vec4(color, 1.0);
}
