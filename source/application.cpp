#include "application.hpp"

#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>
#include <stb_image.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace application {

struct Mat4 { float m[16]; };

static constexpr float PI = 3.14159265358979f;

static Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}

static Mat4 translation(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

static Mat4 scaling(float x, float y, float z) {
    Mat4 r = identity();
    r.m[0] = x;
    r.m[5] = y;
    r.m[10] = z;
    return r;
}

static Mat4 rotationX(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

static Mat4 rotationY(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c;
    r.m[2] = -s;
    r.m[8] = s;
    r.m[10] = c;
    return r;
}

static Mat4 rotationZ(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c;
    r.m[1] = s;
    r.m[4] = -s;
    r.m[5] = c;
    return r;
}

static Mat4 perspective(float fovy, float aspect, float zn, float zf) {
    Mat4 r{};
    float f = 1.0f / std::tan(fovy * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = -f;
    r.m[10] = zf / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = (zn * zf) / (zn - zf);
    return r;
}

static Mat4 ortho(float l, float r_, float b, float t, float zn, float zf) {
    Mat4 r = identity();
    r.m[0] = 2.0f / (r_ - l);
    r.m[5] = 2.0f / (b - t);
    r.m[10] = -1.0f / (zf - zn);
    r.m[12] = -(r_ + l) / (r_ - l);
    r.m[13] = -(b + t) / (b - t);
    r.m[14] = -zn / (zf - zn);
    return r;
}

struct Vec3 { float x, y, z; };

static Vec3 toVec3(const float v[3]) {
    return {v[0], v[1], v[2]};
}

static Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vec3 operator*(Vec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

static float length(Vec3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static Vec3 normalize(Vec3 v) {
    float len = length(v);
    return len > 1e-6f ? v * (1.0f / len) : Vec3{0.0f, 0.0f, 0.0f};
}

static Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct Vertex {
    float pos[3];
    float normal[3];
    float uv[2];
};

struct GizmoVertex {
    Vec3 pos;
    Vec3 color;
};

struct SceneData {
    Mat4 viewProj;
    Vec3 cameraPos;
    uint32_t pointCount;
    Vec3 ambientColor;
    float ambientIntensity;
    Vec3 dirDirection;
    uint32_t spotCount;
    Vec3 dirColor;
    float dirIntensity;
};

struct ModelData {
    Mat4 model;
    Vec3 specular;
    float shininess;
    float uvScale[2];
    float padding[2];
};

struct SwirlData {
    uint32_t enabled;
    float time;
    float strength;
    float radius;
};

struct PointLightData {
    Vec3 position;
    float intensity;
    Vec3 color;
    float padding;
};

struct SpotLightData {
    Vec3 position;
    float intensity;
    Vec3 direction;
    float cosInner;
    Vec3 color;
    float cosOuter;
};

static_assert(sizeof(SceneData) == 128);
static_assert(sizeof(ModelData) == 96);
static_assert(sizeof(PointLightData) == 32);
static_assert(sizeof(SpotLightData) == 48);
static_assert(sizeof(SwirlData) == 16);

struct Material {
    float specular[3];
    float shininess;
    int texture;
    int tiling[2];
};

struct AppState {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float rotation[3] = {0.0f, 0.0f, 0.0f};
    float scale[3] = {1.0f, 1.0f, 1.0f};

    int projection = 0;
    bool animate = true;
    float animSpeed = 1.0f;
    float animRadius = 1.2f;
    float animDepth = 2.0f;
    float animPhase = 0.0f;
    int animFreq = 2;
    bool autoRotate = true;
    bool paused = false;

    Material material = {{0.6f, 0.6f, 0.6f}, 32.0f, 0, {2, 1}};
};

struct SwirlState {
    bool enabled = false;
    float strength = 4.0f;
    float radius = 0.45f;
    float speed = 1.0f;
    float time = 0.0f;
};

struct CameraState {
    float position[3] = {0.0f, 2.5f, 9.0f};
    float rotation[3] = {-12.0f, 0.0f, 0.0f};
    float fov = 45.0f;
    float zNear = 0.1f;
    float zFar = 100.0f;
};

struct PointLightState {
    float position[3];
    float color[3];
    float intensity;
};

struct SpotLightState {
    float position[3];
    float direction[3];
    float color[3];
    float intensity;
    float innerAngle;
    float outerAngle;
};

struct LightingState {
    float ambientColor[3] = {1.0f, 1.0f, 1.0f};
    float ambientIntensity = 0.1f;

    float dirDirection[3] = {0.4f, -1.0f, -0.6f};
    float dirColor[3] = {1.0f, 0.95f, 0.85f};
    float dirIntensity = 0.5f;

    std::vector<PointLightState> points = {
        {{-2.0f, 2.0f, 2.5f}, {1.0f, 0.55f, 0.25f}, 6.0f},
    };
    std::vector<SpotLightState> spots = {
        {{1.5f, 3.5f, 0.0f}, {-0.3f, -1.0f, 0.0f}, {0.35f, 0.65f, 1.0f}, 20.0f, 15.0f, 25.0f},
    };

    bool showGizmos = true;
};

struct FloorState {
    Material material = {{0.25f, 0.25f, 0.25f}, 16.0f, 1, {1, 1}};
    float height = -2.5f;
    float size = 6.0f;
};

struct MeshBuffers {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexAlloc = VK_NULL_HANDLE;

    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VmaAllocation indexAlloc = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

struct Texture {
    std::string name;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct GpuResources {
    MeshBuffers sphere;
    MeshBuffers floor;

    VkCommandPool uploadPool = VK_NULL_HANDLE;
    VkCommandBuffer uploadCmd = VK_NULL_HANDLE;
    VkFence uploadFence = VK_NULL_HANDLE;
    Buffer staging;
    VkDeviceSize stagingOffset = 0;

    std::vector<Texture> textures;
    VkSampler sampler = VK_NULL_HANDLE;

    Buffer sceneUbo;
    Buffer modelUbo;
    VkDeviceSize modelStride = 0;
    Buffer pointLights;
    Buffer spotLights;
    Buffer gizmoVertices;
    uint32_t gizmoTriangleVertices = 0;
    uint32_t gizmoLineVertices = 0;

    VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet sceneSet = VK_NULL_HANDLE;

    VkPipelineLayout modelPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout gizmoPipelineLayout = VK_NULL_HANDLE;
    VkPipeline modelPipeline = VK_NULL_HANDLE;
    VkPipeline gizmoTrianglePipeline = VK_NULL_HANDLE;
    VkPipeline gizmoLinePipeline = VK_NULL_HANDLE;
};

struct PendingFrame {
    SceneData scene{};
    ModelData sphere{};
    ModelData floor{};
    SwirlData swirl{};
    int sphereTexture = 0;
    int floorTexture = 0;
    std::vector<PointLightData> points;
    std::vector<SpotLightData> spots;
    std::vector<GizmoVertex> gizmos;
    uint32_t gizmoTriangleVertices = 0;
};

static constexpr size_t initialLightCapacity = 16;
static constexpr size_t initialGizmoCapacity = 1024;
static constexpr float gizmoSphereRadius = 0.1f;
static constexpr float gizmoConeLength = 1.2f;
static constexpr VkDeviceSize stagingSize = 64 * 1024 * 1024;
static constexpr const char* texturesDir = "textures";

static AppState state;
static SwirlState swirl;
static CameraState camera;
static LightingState lighting;
static FloorState floorState;
static GpuResources gpu;
static PendingFrame pending;
static std::vector<Vec3> gizmoSphere;
static double lastTime = 0.0;

static void generateSphere(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, int stacks, int slices, float radius) {
    verts.clear();
    indices.clear();

    for (int i = 0; i <= stacks; ++i) {
        float phi = PI * static_cast<float>(i) / static_cast<float>(stacks);
        float y = std::cos(phi) * radius;
        float r = std::sin(phi) * radius;

        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
            float x = r * std::cos(theta);
            float z = r * std::sin(theta);

            Vertex v{};
            v.pos[0] = x;
            v.pos[1] = y;
            v.pos[2] = z;
            v.normal[0] = x / radius;
            v.normal[1] = y / radius;
            v.normal[2] = z / radius;
            v.uv[0] = 1.0f - static_cast<float>(j) / static_cast<float>(slices);
            v.uv[1] = static_cast<float>(i) / static_cast<float>(stacks);
            verts.push_back(v);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = static_cast<uint32_t>(i * (slices + 1) + j);
            uint32_t b = a + static_cast<uint32_t>(slices + 1);

            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(a + 1);

            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b + 1);
        }
    }
}

static void generateFloor(std::vector<Vertex>& verts, std::vector<uint32_t>& indices) {
    verts = {
        {{-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        {{-1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        {{1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
    };
    indices = {0, 1, 2, 1, 3, 2};
}

static void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& alloc, void** mapped = nullptr) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai, &buffer, &alloc, &info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }
    if (mapped) *mapped = info.pMappedData;
}

static void createBuffer(Buffer& buf, VkDeviceSize size, VkBufferUsageFlags usage) {
    createBuffer(size, usage, buf.buffer, buf.alloc, &buf.mapped);
    buf.size = size;
}

static void createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& alloc) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai, &buffer, &alloc, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create device buffer");
    }
}

static void createDeviceBuffer(Buffer& buf, VkDeviceSize size, VkBufferUsageFlags usage) {
    createDeviceBuffer(size, usage, buf.buffer, buf.alloc);
    buf.mapped = nullptr;
    buf.size = size;
}

static void destroyBuffer(Buffer& buf) {
    vmaDestroyBuffer(graphics::internal::context.allocator, buf.buffer, buf.alloc);
    buf = Buffer{};
}

static bool ensureSize(Buffer& buf, VkDeviceSize size, VkBufferUsageFlags usage, bool deviceLocal = false) {
    if (size <= buf.size) return false;

    Buffer bigger;
    if (deviceLocal) {
        createDeviceBuffer(bigger, std::max(size, buf.size * 2), usage);
    } else {
        createBuffer(bigger, std::max(size, buf.size * 2), usage);
    }
    destroyBuffer(buf);
    buf = bigger;
    return true;
}

static void upload(Buffer& buf, const void* data, VkDeviceSize size, VkDeviceSize offset = 0) {
    if (size == 0) return;
    std::memcpy(static_cast<char*>(buf.mapped) + offset, data, size);
    vmaFlushAllocation(graphics::internal::context.allocator, buf.alloc, offset, size);
}

static void createUploader() {
    auto& ctx = graphics::internal::context;

    VkCommandPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = ctx.graphics_queue_index;
    if (vkCreateCommandPool(ctx.device, &pi, nullptr, &gpu.uploadPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create upload command pool");
    }

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = gpu.uploadPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx.device, &ai, &gpu.uploadCmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate upload command buffer");
    }

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(ctx.device, &fi, nullptr, &gpu.uploadFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create upload fence");
    }

    createBuffer(gpu.staging, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
}

static void beginUpload() {
    gpu.stagingOffset = 0;
    vkResetCommandBuffer(gpu.uploadCmd, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(gpu.uploadCmd, &bi);
}

static VkDeviceSize stage(const void* data, VkDeviceSize size) {
    VkDeviceSize offset = (gpu.stagingOffset + 15) / 16 * 16;
    if (offset + size > gpu.staging.size) {
        throw std::runtime_error("Staging buffer is too small");
    }
    upload(gpu.staging, data, size, offset);
    gpu.stagingOffset = offset + size;
    return offset;
}

static void uploadBuffer(VkBuffer dst, const void* data, VkDeviceSize size) {
    if (size == 0) return;

    VkBufferCopy region{};
    region.srcOffset = stage(data, size);
    region.dstOffset = 0;
    region.size = size;
    vkCmdCopyBuffer(gpu.uploadCmd, gpu.staging.buffer, dst, 1, &region);

    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = dst;
    barrier.offset = 0;
    barrier.size = size;
    vkCmdPipelineBarrier(gpu.uploadCmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
}

static void uploadImage(VkImage image, uint32_t width, uint32_t height, const void* pixels) {
    VkDeviceSize offset = stage(pixels, static_cast<VkDeviceSize>(width) * height * 4);

    VkImageMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = 0;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = image;
    toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(gpu.uploadCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toTransfer);

    VkBufferImageCopy region{};
    region.bufferOffset = offset;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(gpu.uploadCmd, gpu.staging.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier toShader = toTransfer;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(gpu.uploadCmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toShader);
}

static void endUpload() {
    auto& ctx = graphics::internal::context;
    vkEndCommandBuffer(gpu.uploadCmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &gpu.uploadCmd;

    vkResetFences(ctx.device, 1, &gpu.uploadFence);
    if (vkQueueSubmit(ctx.graphics_queue, 1, &si, gpu.uploadFence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit upload commands");
    }
    vkWaitForFences(ctx.device, 1, &gpu.uploadFence, VK_TRUE, UINT64_MAX);
}

static void destroyUploader() {
    auto& ctx = graphics::internal::context;
    destroyBuffer(gpu.staging);
    vkDestroyFence(ctx.device, gpu.uploadFence, nullptr);
    vkDestroyCommandPool(ctx.device, gpu.uploadPool, nullptr);
}

static void createMesh(MeshBuffers& mesh, const std::vector<Vertex>& verts, const std::vector<uint32_t>& indices) {
    mesh.indexCount = static_cast<uint32_t>(indices.size());

    VkDeviceSize vertexSize = verts.size() * sizeof(Vertex);
    VkDeviceSize indexSize = indices.size() * sizeof(uint32_t);
    createDeviceBuffer(vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertexBuffer, mesh.vertexAlloc);
    createDeviceBuffer(indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indexBuffer, mesh.indexAlloc);

    uploadBuffer(mesh.vertexBuffer, verts.data(), vertexSize);
    uploadBuffer(mesh.indexBuffer, indices.data(), indexSize);
}

static void destroyMesh(MeshBuffers& mesh) {
    auto& ctx = graphics::internal::context;
    vmaDestroyBuffer(ctx.allocator, mesh.indexBuffer, mesh.indexAlloc);
    vmaDestroyBuffer(ctx.allocator, mesh.vertexBuffer, mesh.vertexAlloc);
}

static void createImage(Texture& tex, uint32_t width, uint32_t height) {
    auto& ctx = graphics::internal::context;

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_SRGB;
    ii.extent = {width, height, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateImage(ctx.allocator, &ii, &ai, &tex.image, &tex.alloc, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = tex.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_SRGB;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(ctx.device, &vi, nullptr, &tex.view) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view");
    }
}

static bool isImageFile(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga" || ext == ".gif";
}

static void loadTextures() {
    std::vector<std::filesystem::path> paths;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(texturesDir, ec)) {
        if (entry.is_regular_file() && isImageFile(entry.path())) paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());

    for (const auto& path : paths) {
        int width = 0, height = 0, channels = 0;
        stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!pixels) {
            std::cerr << "Cannot load " << path.string() << ": " << stbi_failure_reason() << '\n';
            continue;
        }
        if (static_cast<VkDeviceSize>(width) * height * 4 > stagingSize) {
            std::cerr << "Image is too large: " << path.string() << '\n';
            stbi_image_free(pixels);
            continue;
        }

        Texture tex;
        tex.name = path.filename().string();
        createImage(tex, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

        beginUpload();
        uploadImage(tex.image, static_cast<uint32_t>(width), static_cast<uint32_t>(height), pixels);
        endUpload();
        stbi_image_free(pixels);

        gpu.textures.push_back(tex);
    }

    if (gpu.textures.empty()) {
        throw std::runtime_error("No images found in textures/");
    }
    floorState.material.texture = std::min(floorState.material.texture, static_cast<int>(gpu.textures.size()) - 1);

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = 0.0f;

    if (vkCreateSampler(graphics::internal::context.device, &si, nullptr, &gpu.sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler");
    }
}

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    size_t sz = static_cast<size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    return mod;
}

static VkDescriptorSetLayout createSetLayout(const VkDescriptorSetLayoutBinding* bindings, uint32_t count) {
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = count;
    ci.pBindings = bindings;

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(graphics::internal::context.device, &ci, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    return layout;
}

static void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buf,
                            VkDeviceSize range = VK_WHOLE_SIZE) {
    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = buf.buffer;
    bufInfo.offset = 0;
    bufInfo.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &bufInfo;

    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

static void writeImageDescriptor(VkDescriptorSet set, uint32_t binding, VkImageView view) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = gpu.sampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

static VkPipelineLayout createPipelineLayout(const VkDescriptorSetLayout* setLayouts, uint32_t count,
                                             const VkPushConstantRange* pushRange = nullptr) {
    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = count;
    ci.pSetLayouts = setLayouts;
    ci.pushConstantRangeCount = pushRange ? 1 : 0;
    ci.pPushConstantRanges = pushRange;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(graphics::internal::context.device, &ci, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
    return layout;
}

static VkPipeline createPipeline(const std::string& vertPath, const std::string& fragPath,
                                 uint32_t stride, const VkVertexInputAttributeDescription* attrs, uint32_t attrCount,
                                 VkPrimitiveTopology topology, VkCullModeFlags cullMode, VkPipelineLayout layout) {
    auto& ctx = graphics::internal::context;

    auto vertCode = readFile(vertPath);
    auto fragCode = readFile(fragPath);

    VkShaderModule vertMod = createShaderModule(ctx.device, vertCode);
    VkShaderModule fragMod = createShaderModule(ctx.device, fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = stride;
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bind;
    vertexInput.vertexAttributeDescriptionCount = attrCount;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAsm{};
    inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAsm.topology = topology;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = cullMode;
    raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAtt;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vertexInput;
    ci.pInputAssemblyState = &inputAsm;
    ci.pViewportState = &viewportState;
    ci.pRasterizationState = &raster;
    ci.pMultisampleState = &multisample;
    ci.pDepthStencilState = &depthStencil;
    ci.pColorBlendState = &blend;
    ci.pDynamicState = &dynamic;
    ci.layout = layout;
    ci.renderPass = ctx.render_pass;
    ci.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }

    vkDestroyShaderModule(ctx.device, vertMod, nullptr);
    vkDestroyShaderModule(ctx.device, fragMod, nullptr);
    return pipeline;
}

static Vec3 brightColor(Vec3 c) {
    float m = std::max({c.x, c.y, c.z});
    return m > 1e-4f ? c * (1.0f / m) : c;
}

static void appendLightSphere(std::vector<GizmoVertex>& out, Vec3 center, Vec3 color) {
    for (Vec3 p : gizmoSphere) {
        out.push_back({center + p, color});
    }
}

static void appendCone(std::vector<GizmoVertex>& out, Vec3 apex, Vec3 dir, float angle, float len, Vec3 color) {
    const int circlePoints = 24;
    const int sideLines = 8;

    Vec3 axis = normalize(dir);
    Vec3 helper = std::fabs(axis.y) < 0.99f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    Vec3 side = normalize(cross(axis, helper));
    Vec3 up = cross(axis, side);

    Vec3 base = apex + axis * (len * std::cos(angle));
    float radius = len * std::sin(angle);

    auto circle = [&](int k) {
        float t = 2.0f * PI * static_cast<float>(k) / static_cast<float>(circlePoints);
        return base + side * (radius * std::cos(t)) + up * (radius * std::sin(t));
    };

    for (int k = 0; k < circlePoints; ++k) {
        out.push_back({circle(k), color});
        out.push_back({circle(k + 1), color});
    }

    for (int k = 0; k < circlePoints; k += circlePoints / sideLines) {
        out.push_back({apex, color});
        out.push_back({circle(k), color});
    }

    out.push_back({apex, color});
    out.push_back({apex + axis * len, color});
}

static ModelData modelData(const Mat4& model, const Material& material) {
    ModelData d{};
    d.model = model;
    d.specular = toVec3(material.specular);
    d.shininess = material.shininess;
    d.uvScale[0] = static_cast<float>(material.tiling[0]);
    d.uvScale[1] = static_cast<float>(material.tiling[1]);
    return d;
}

static void drawMesh(VkCommandBuffer cmd, const MeshBuffers& mesh, int texture, uint32_t dynamicOffset,
                     const SwirlData& effect) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.modelPipelineLayout, 1, 1,
                            &gpu.textures[texture].set, 1, &dynamicOffset);
    vkCmdPushConstants(cmd, gpu.modelPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SwirlData), &effect);
    vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
}

static void editMaterial(Material& material) {
    if (ImGui::BeginCombo("Texture", gpu.textures[material.texture].name.c_str())) {
        for (int i = 0; i < static_cast<int>(gpu.textures.size()); ++i) {
            bool selected = i == material.texture;
            if (ImGui::Selectable(gpu.textures[i].name.c_str(), selected)) material.texture = i;
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SliderInt2("Tiling", material.tiling, 1, 8, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::ColorEdit3("Specular", material.specular);
    ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 256.0f, "%.0f",
                       ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
}

static void sceneWindow() {
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 10.0f, 10.0f), ImGuiCond_FirstUseEver, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(380.0f, 650.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Scene");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("camera");
        ImGui::RadioButton("Perspective", &state.projection, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &state.projection, 1);
        ImGui::SliderFloat3("Position", camera.position, -20.0f, 20.0f);
        ImGui::SliderFloat3("Rotation, deg", camera.rotation, -180.0f, 180.0f);
        ImGui::SliderFloat("FOV, deg", &camera.fov, 20.0f, 120.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Near", &camera.zNear, 0.01f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Far", &camera.zFar, 10.0f, 200.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::Button("Reset camera")) camera = CameraState{};
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("ambient");
        ImGui::ColorEdit3("Color", lighting.ambientColor);
        ImGui::SliderFloat("Intensity", &lighting.ambientIntensity, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("directional");
        ImGui::SliderFloat3("Direction", lighting.dirDirection, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", lighting.dirColor);
        ImGui::SliderFloat("Intensity", &lighting.dirIntensity, 0.0f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("point");
        if (ImGui::Button("Add point light")) {
            lighting.points.push_back({{0.0f, 3.5f, 2.0f}, {1.0f, 1.0f, 1.0f}, 6.0f});
        }
        int removeIndex = -1;
        for (int i = 0; i < static_cast<int>(lighting.points.size()); ++i) {
            PointLightState& p = lighting.points[i];
            ImGui::PushID(i);
            if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Point light %d", i)) {
                ImGui::SliderFloat3("Position", p.position, -10.0f, 10.0f);
                ImGui::ColorEdit3("Color", p.color);
                ImGui::SliderFloat("Intensity", &p.intensity, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
                if (ImGui::Button("Remove")) removeIndex = i;
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (removeIndex >= 0) lighting.points.erase(lighting.points.begin() + removeIndex);
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("spot");
        if (ImGui::Button("Add spot light")) {
            lighting.spots.push_back({{0.0f, 3.5f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 20.0f, 15.0f, 25.0f});
        }
        int removeIndex = -1;
        for (int i = 0; i < static_cast<int>(lighting.spots.size()); ++i) {
            SpotLightState& s = lighting.spots[i];
            ImGui::PushID(i);
            if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Spot light %d", i)) {
                ImGui::SliderFloat3("Position", s.position, -10.0f, 10.0f);
                ImGui::SliderFloat3("Direction", s.direction, -1.0f, 1.0f);
                ImGui::ColorEdit3("Color", s.color);
                ImGui::SliderFloat("Intensity", &s.intensity, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
                ImGui::SliderFloat("Inner angle", &s.innerAngle, 0.0f, 85.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
                ImGui::SliderFloat("Outer angle", &s.outerAngle, 0.0f, 85.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
                if (ImGui::Button("Remove")) removeIndex = i;
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (removeIndex >= 0) lighting.spots.erase(lighting.spots.begin() + removeIndex);
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Light sources", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show light sources", &lighting.showGizmos);
    }

    if (ImGui::CollapsingHeader("Floor")) {
        ImGui::PushID("floor");
        editMaterial(floorState.material);
        ImGui::SliderFloat("Height", &floorState.height, -5.0f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Size", &floorState.size, 1.0f, 20.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopID();
    }

    ImGui::End();
}

bool initialize() {
    auto& ctx = graphics::internal::context;

    createUploader();
    loadTextures();

    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    beginUpload();
    generateSphere(verts, indices, 9, 9, 1.0f);
    createMesh(gpu.sphere, verts, indices);

    generateFloor(verts, indices);
    createMesh(gpu.floor, verts, indices);
    endUpload();

    generateSphere(verts, indices, 6, 8, gizmoSphereRadius);
    for (uint32_t i : indices) {
        gizmoSphere.push_back(toVec3(verts[i].pos));
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
    VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
    gpu.modelStride = (sizeof(ModelData) + align - 1) / align * align;

    createBuffer(gpu.sceneUbo, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    createBuffer(gpu.modelUbo, gpu.modelStride * 2, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    createBuffer(gpu.pointLights, initialLightCapacity * sizeof(PointLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    createBuffer(gpu.spotLights, initialLightCapacity * sizeof(SpotLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    createDeviceBuffer(gpu.gizmoVertices, initialGizmoCapacity * sizeof(GizmoVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    {
        VkDescriptorSetLayoutBinding sceneBindings[3]{};
        sceneBindings[0].binding = 0;
        sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sceneBindings[0].descriptorCount = 1;
        sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        sceneBindings[1].binding = 1;
        sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sceneBindings[1].descriptorCount = 1;
        sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        sceneBindings[2].binding = 2;
        sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sceneBindings[2].descriptorCount = 1;
        sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        gpu.sceneSetLayout = createSetLayout(sceneBindings, 3);

        VkDescriptorSetLayoutBinding materialBindings[2]{};
        materialBindings[0].binding = 0;
        materialBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        materialBindings[0].descriptorCount = 1;
        materialBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        materialBindings[1].binding = 1;
        materialBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        materialBindings[1].descriptorCount = 1;
        materialBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        gpu.materialSetLayout = createSetLayout(materialBindings, 2);
    }

    {
        uint32_t textureCount = static_cast<uint32_t>(gpu.textures.size());

        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 2;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        poolSizes[2].descriptorCount = textureCount;
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[3].descriptorCount = textureCount;

        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 4;
        ci.pPoolSizes = poolSizes;
        ci.maxSets = 1 + textureCount;

        if (vkCreateDescriptorPool(ctx.device, &ci, nullptr, &gpu.descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        std::vector<VkDescriptorSetLayout> layouts(1 + textureCount, gpu.materialSetLayout);
        layouts[0] = gpu.sceneSetLayout;
        std::vector<VkDescriptorSet> sets(layouts.size());

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = gpu.descriptorPool;
        ai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        ai.pSetLayouts = layouts.data();

        if (vkAllocateDescriptorSets(ctx.device, &ai, sets.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor sets");
        }

        gpu.sceneSet = sets[0];
        writeDescriptor(gpu.sceneSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, gpu.sceneUbo);
        writeDescriptor(gpu.sceneSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gpu.pointLights);
        writeDescriptor(gpu.sceneSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gpu.spotLights);

        for (uint32_t i = 0; i < textureCount; ++i) {
            Texture& tex = gpu.textures[i];
            tex.set = sets[1 + i];
            writeDescriptor(tex.set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, gpu.modelUbo, sizeof(ModelData));
            writeImageDescriptor(tex.set, 1, tex.view);
        }
    }

    {
        VkPushConstantRange swirlRange{};
        swirlRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        swirlRange.offset = 0;
        swirlRange.size = sizeof(SwirlData);

        VkDescriptorSetLayout modelLayouts[2] = {gpu.sceneSetLayout, gpu.materialSetLayout};
        gpu.modelPipelineLayout = createPipelineLayout(modelLayouts, 2, &swirlRange);
        gpu.gizmoPipelineLayout = createPipelineLayout(&gpu.sceneSetLayout, 1);
    }

    {
        VkVertexInputAttributeDescription attrs[3]{};
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].location = 1;
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, normal);
        attrs[2].location = 2;
        attrs[2].binding = 0;
        attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
        attrs[2].offset = offsetof(Vertex, uv);

        gpu.modelPipeline = createPipeline("shaders/model.vert.spv", "shaders/model.frag.spv",
                                           sizeof(Vertex), attrs, 3,
                                           VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_BACK_BIT,
                                           gpu.modelPipelineLayout);
    }

    {
        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(GizmoVertex, pos);
        attrs[1].location = 1;
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(GizmoVertex, color);

        gpu.gizmoTrianglePipeline = createPipeline("shaders/gizmo.vert.spv", "shaders/gizmo.frag.spv",
                                                   sizeof(GizmoVertex), attrs, 2,
                                                   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_NONE,
                                                   gpu.gizmoPipelineLayout);
        gpu.gizmoLinePipeline = createPipeline("shaders/gizmo.vert.spv", "shaders/gizmo.frag.spv",
                                               sizeof(GizmoVertex), attrs, 2,
                                               VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE,
                                               gpu.gizmoPipelineLayout);
    }

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    vkDestroyPipeline(ctx.device, gpu.gizmoLinePipeline, nullptr);
    vkDestroyPipeline(ctx.device, gpu.gizmoTrianglePipeline, nullptr);
    vkDestroyPipeline(ctx.device, gpu.modelPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, gpu.gizmoPipelineLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, gpu.modelPipelineLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, gpu.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, gpu.materialSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, gpu.sceneSetLayout, nullptr);

    vkDestroySampler(ctx.device, gpu.sampler, nullptr);
    for (Texture& tex : gpu.textures) {
        vkDestroyImageView(ctx.device, tex.view, nullptr);
        vmaDestroyImage(ctx.allocator, tex.image, tex.alloc);
    }
    gpu.textures.clear();

    destroyBuffer(gpu.gizmoVertices);
    destroyBuffer(gpu.spotLights);
    destroyBuffer(gpu.pointLights);
    destroyBuffer(gpu.modelUbo);
    destroyBuffer(gpu.sceneUbo);
    destroyMesh(gpu.floor);
    destroyMesh(gpu.sphere);
    destroyUploader();
}

void update(double time) {
    double dt = (lastTime == 0.0) ? 0.0 : (time - lastTime);
    lastTime = time;

    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Sphere");

    ImGui::Text("Transform");
    ImGui::SliderFloat3("Position", state.position, -5.0f, 5.0f);
    ImGui::SliderFloat3("Rotation, deg", state.rotation, -180.0f, 180.0f);
    ImGui::SliderFloat3("Scale", state.scale, 0.1f, 3.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);

    ImGui::Separator();
    ImGui::Text("Animation");
    ImGui::Checkbox("Animate", &state.animate);
    ImGui::SameLine();
    if (ImGui::Button(state.paused ? "Play" : "Pause")) state.paused = !state.paused;

    ImGui::SliderFloat("Speed", &state.animSpeed, 0.1f, 5.0f);
    ImGui::SliderFloat("Radius", &state.animRadius, 0.0f, 3.0f);
    ImGui::SliderFloat("Depth", &state.animDepth, 0.0f, 3.0f);
    ImGui::SliderInt("Frequency", &state.animFreq, 1, 5);
    ImGui::Checkbox("Auto-rotate", &state.autoRotate);
    ImGui::Separator();
    ImGui::Text("Material");
    editMaterial(state.material);
    ImGui::Separator();
    ImGui::Text("Swirl");
    ImGui::Checkbox("Enable swirl", &swirl.enabled);
    ImGui::SliderFloat("Twist", &swirl.strength, 0.0f, 10.0f, "%.2f rad", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderFloat("Swirl radius", &swirl.radius, 0.05f, 0.5f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderFloat("Swirl speed", &swirl.speed, 0.0f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);

    ImGui::End();

    sceneWindow();

    static float spinPhase = 0.0f;
    if (!state.paused) {
        if (state.animate) state.animPhase += static_cast<float>(dt) * state.animSpeed;
        if (state.animPhase > 2.0f * PI) state.animPhase -= 2.0f * PI;
        if (state.autoRotate) spinPhase += static_cast<float>(dt) * state.animSpeed;
    }
    swirl.time += static_cast<float>(dt) * swirl.speed;

    float animX = 0.0f, animY = 0.0f, animZ = 0.0f;
    if (state.animate) {
        float p = static_cast<float>(state.animFreq);
        animX = state.animRadius * std::sin(p * state.animPhase);
        animY = state.animRadius * std::sin((p + 1.0f) * state.animPhase);
        animZ = state.animDepth * std::sin(state.animPhase);
    }

    float autoRotX = spinPhase * 0.5f;
    float autoRotY = spinPhase * 0.7f;
    float autoRotZ = spinPhase * 0.3f;

    Mat4 model = identity();
    model = multiply(model, translation(state.position[0] + animX, state.position[1] + animY, state.position[2] + animZ));
    model = multiply(model, rotationY(state.rotation[1] * PI / 180.0f + autoRotY));
    model = multiply(model, rotationX(state.rotation[0] * PI / 180.0f + autoRotX));
    model = multiply(model, rotationZ(state.rotation[2] * PI / 180.0f + autoRotZ));
    model = multiply(model, scaling(state.scale[0], state.scale[1], state.scale[2]));

    Mat4 floorModel = multiply(translation(0.0f, floorState.height, 0.0f), scaling(floorState.size, 1.0f, floorState.size));

    Mat4 view = rotationZ(-camera.rotation[2] * PI / 180.0f);
    view = multiply(view, rotationX(-camera.rotation[0] * PI / 180.0f));
    view = multiply(view, rotationY(-camera.rotation[1] * PI / 180.0f));
    view = multiply(view, translation(-camera.position[0], -camera.position[1], -camera.position[2]));

    const float fovY = camera.fov * PI / 180.0f;

    auto& ctx = graphics::internal::context;
    float aspect = static_cast<float>(ctx.swapchain_extent.width) / static_cast<float>(ctx.swapchain_extent.height);

    Mat4 proj;
    if (state.projection == 0) {
        proj = perspective(fovY, aspect, camera.zNear, camera.zFar);
    } else {
        float h = length(toVec3(CameraState{}.position)) * std::tan(fovY * 0.5f);
        float w = h * aspect;
        proj = ortho(-w, w, -h, h, camera.zNear, camera.zFar);
    }

    pending.points.clear();
    for (const PointLightState& p : lighting.points) {
        pending.points.push_back({toVec3(p.position), p.intensity, toVec3(p.color), 0.0f});
    }

    pending.spots.clear();
    for (const SpotLightState& s : lighting.spots) {
        pending.spots.push_back({toVec3(s.position), s.intensity, normalize(toVec3(s.direction)),
                                 std::cos(s.innerAngle * PI / 180.0f), toVec3(s.color),
                                 std::cos(s.outerAngle * PI / 180.0f)});
    }

    pending.gizmos.clear();
    if (lighting.showGizmos) {
        for (const PointLightState& p : lighting.points) {
            appendLightSphere(pending.gizmos, toVec3(p.position), brightColor(toVec3(p.color)));
        }
        for (const SpotLightState& s : lighting.spots) {
            appendLightSphere(pending.gizmos, toVec3(s.position), brightColor(toVec3(s.color)));
        }
    }
    pending.gizmoTriangleVertices = static_cast<uint32_t>(pending.gizmos.size());
    if (lighting.showGizmos) {
        for (const SpotLightState& s : lighting.spots) {
            appendCone(pending.gizmos, toVec3(s.position), toVec3(s.direction), s.outerAngle * PI / 180.0f,
                       gizmoConeLength, Vec3{1.0f, 1.0f, 1.0f});
        }
    }

    pending.scene.viewProj = multiply(proj, view);
    pending.scene.cameraPos = toVec3(camera.position);
    pending.scene.pointCount = static_cast<uint32_t>(pending.points.size());
    pending.scene.ambientColor = toVec3(lighting.ambientColor);
    pending.scene.ambientIntensity = lighting.ambientIntensity;
    pending.scene.dirDirection = normalize(toVec3(lighting.dirDirection));
    pending.scene.spotCount = static_cast<uint32_t>(pending.spots.size());
    pending.scene.dirColor = toVec3(lighting.dirColor);
    pending.scene.dirIntensity = lighting.dirIntensity;

    pending.sphere = modelData(model, state.material);
    pending.floor = modelData(floorModel, floorState.material);
    pending.sphereTexture = state.material.texture;
    pending.floorTexture = floorState.material.texture;
    pending.swirl = {swirl.enabled ? 1u : 0u, swirl.time, swirl.strength, swirl.radius};
}

void render(const graphics::internal::FrameData& fd) {
    if (fd.command_buffer == VK_NULL_HANDLE) return;

    auto& ctx = graphics::internal::context;
    VkCommandBuffer cmd = fd.command_buffer;

    if (ensureSize(gpu.pointLights, pending.points.size() * sizeof(PointLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        writeDescriptor(gpu.sceneSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gpu.pointLights);
    }
    if (ensureSize(gpu.spotLights, pending.spots.size() * sizeof(SpotLightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        writeDescriptor(gpu.sceneSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gpu.spotLights);
    }
    ensureSize(gpu.gizmoVertices, pending.gizmos.size() * sizeof(GizmoVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true);

    upload(gpu.sceneUbo, &pending.scene, sizeof(SceneData));
    upload(gpu.modelUbo, &pending.sphere, sizeof(ModelData), 0);
    upload(gpu.modelUbo, &pending.floor, sizeof(ModelData), gpu.modelStride);
    upload(gpu.pointLights, pending.points.data(), pending.points.size() * sizeof(PointLightData));
    upload(gpu.spotLights, pending.spots.data(), pending.spots.size() * sizeof(SpotLightData));
    if (!pending.gizmos.empty()) {
        beginUpload();
        uploadBuffer(gpu.gizmoVertices.buffer, pending.gizmos.data(), pending.gizmos.size() * sizeof(GizmoVertex));
        endUpload();
    }
    gpu.gizmoTriangleVertices = pending.gizmoTriangleVertices;
    gpu.gizmoLineVertices = static_cast<uint32_t>(pending.gizmos.size()) - pending.gizmoTriangleVertices;

    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkClearValue clearValues[2]{};
    clearValues[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = ctx.render_pass;
    rpBegin.framebuffer = fd.framebuffer;
    rpBegin.renderArea = {{0, 0}, ctx.swapchain_extent};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;
    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.modelPipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(ctx.swapchain_extent.width);
    viewport.height = static_cast<float>(ctx.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.modelPipelineLayout, 0, 1, &gpu.sceneSet, 0, nullptr);
    drawMesh(cmd, gpu.sphere, pending.sphereTexture, 0, pending.swirl);
    drawMesh(cmd, gpu.floor, pending.floorTexture, static_cast<uint32_t>(gpu.modelStride), SwirlData{});

    if (gpu.gizmoTriangleVertices + gpu.gizmoLineVertices > 0) {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &gpu.gizmoVertices.buffer, &offset);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.gizmoTrianglePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.gizmoPipelineLayout, 0, 1, &gpu.sceneSet, 0, nullptr);
        if (gpu.gizmoTriangleVertices > 0) vkCmdDraw(cmd, gpu.gizmoTriangleVertices, 1, 0, 0);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.gizmoLinePipeline);
        if (gpu.gizmoLineVertices > 0) vkCmdDraw(cmd, gpu.gizmoLineVertices, 1, gpu.gizmoTriangleVertices, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application
