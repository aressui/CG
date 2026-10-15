#include "application.hpp"

#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

#include <cmath>
#include <cstring>
#include <fstream>
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

struct Vertex {
    float pos[3];
    float normal[3];
    float color[3];
};

struct UniformData {
    Mat4 mvp;
    Mat4 model;
    float userColor[4];
};

struct AppState {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float rotation[3] = {0.0f, 0.0f, 0.0f};
    float scale[3] = {1.0f, 1.0f, 1.0f};

    int projection = 0;
    bool animate = false;
    float animSpeed = 1.0f;
    float animRadius = 2.0f;
    float animPhase = 0.0f;
    float animParam = 1.0f;
    int trajectoryType = 0;
    bool autoRotate = true;
    bool paused = false;

    float userColor[3] = {1.0f, 1.0f, 1.0f};
};

struct GpuResources {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexAlloc = VK_NULL_HANDLE;

    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VmaAllocation indexAlloc = VK_NULL_HANDLE;
    uint32_t indexCount = 0;

    VkBuffer ubo = VK_NULL_HANDLE;
    VmaAllocation uboAlloc = VK_NULL_HANDLE;
    UniformData* uboMapped = nullptr;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

static AppState state;
static GpuResources gpu;
static UniformData pendingUniforms{};
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
            v.color[0] = (x / radius + 1.0f) * 0.5f;
            v.color[1] = (y / radius + 1.0f) * 0.5f;
            v.color[2] = (z / radius + 1.0f) * 0.5f;
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

bool initialize() {
    auto& ctx = graphics::internal::context;

    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    generateSphere(verts, indices, 9, 9, 1.0f);
    gpu.indexCount = static_cast<uint32_t>(indices.size());

    {
        VkDeviceSize size = verts.size() * sizeof(Vertex);
        void* data = nullptr;
        createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, gpu.vertexBuffer, gpu.vertexAlloc, &data);
        std::memcpy(data, verts.data(), size);
        vmaFlushAllocation(ctx.allocator, gpu.vertexAlloc, 0, size);
    }

    {
        VkDeviceSize size = indices.size() * sizeof(uint32_t);
        void* data = nullptr;
        createBuffer(size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, gpu.indexBuffer, gpu.indexAlloc, &data);
        std::memcpy(data, indices.data(), size);
        vmaFlushAllocation(ctx.allocator, gpu.indexAlloc, 0, size);
    }

    {
        void* data = nullptr;
        createBuffer(sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, gpu.ubo, gpu.uboAlloc, &data);
        gpu.uboMapped = static_cast<UniformData*>(data);
        std::memset(gpu.uboMapped, 0, sizeof(UniformData));
    }

    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &binding;

        if (vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &gpu.descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }
    }

    {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &poolSize;
        ci.maxSets = 1;

        if (vkCreateDescriptorPool(ctx.device, &ci, nullptr, &gpu.descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = gpu.descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &gpu.descriptorSetLayout;

        if (vkAllocateDescriptorSets(ctx.device, &ai, &gpu.descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = gpu.ubo;
        bufInfo.offset = 0;
        bufInfo.range = sizeof(UniformData);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = gpu.descriptorSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bufInfo;

        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }

    {
        VkPipelineLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &gpu.descriptorSetLayout;

        if (vkCreatePipelineLayout(ctx.device, &ci, nullptr, &gpu.pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }
    }

    {
        auto vertCode = readFile("shaders/sphere.vert.spv");
        auto fragCode = readFile("shaders/sphere.frag.spv");

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
        bind.stride = sizeof(Vertex);
        bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

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
        attrs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[2].offset = offsetof(Vertex, color);

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &bind;
        vertexInput.vertexAttributeDescriptionCount = 3;
        vertexInput.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo inputAsm{};
        inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAsm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_BACK_BIT;
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
        ci.layout = gpu.pipelineLayout;
        ci.renderPass = ctx.render_pass;
        ci.subpass = 0;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &gpu.pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline");
        }

        vkDestroyShaderModule(ctx.device, vertMod, nullptr);
        vkDestroyShaderModule(ctx.device, fragMod, nullptr);
    }

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    vkDestroyPipeline(ctx.device, gpu.pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, gpu.pipelineLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, gpu.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, gpu.descriptorSetLayout, nullptr);

    vmaDestroyBuffer(ctx.allocator, gpu.ubo, gpu.uboAlloc);
    vmaDestroyBuffer(ctx.allocator, gpu.indexBuffer, gpu.indexAlloc);
    vmaDestroyBuffer(ctx.allocator, gpu.vertexBuffer, gpu.vertexAlloc);
}

void update(double time) {
    double dt = (lastTime == 0.0) ? 0.0 : (time - lastTime);
    lastTime = time;

    ImGui::Begin("Sphere");

    ImGui::Text("Projection");
    ImGui::RadioButton("Perspective", &state.projection, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Orthographic", &state.projection, 1);

    ImGui::Separator();
    ImGui::Text("Transform");
    ImGui::SliderFloat3("Position", state.position, -5.0f, 5.0f);
    ImGui::SliderFloat3("Rotation, deg", state.rotation, -180.0f, 180.0f);
    ImGui::SliderFloat3("Scale", state.scale, 0.1f, 3.0f);

    ImGui::Separator();
    ImGui::Text("Animation");
    ImGui::Checkbox("Animate", &state.animate);
    ImGui::SameLine();
    if (ImGui::Button(state.paused ? "Play" : "Pause")) state.paused = !state.paused;

    static const char* trajectoryNames[] = { "Circle", "Ellipse", "Lissajous", "Spiral" };
    ImGui::Combo("Trajectory", &state.trajectoryType, trajectoryNames, IM_ARRAYSIZE(trajectoryNames));

    ImGui::SliderFloat("Speed", &state.animSpeed, 0.1f, 5.0f);
    ImGui::SliderFloat("Radius", &state.animRadius, 0.0f, 5.0f);
    ImGui::SliderFloat("Param", &state.animParam, 0.1f, 4.0f);
    ImGui::Checkbox("Auto-rotate", &state.autoRotate);
    ImGui::Separator();
    ImGui::ColorEdit3("Color", state.userColor);

    ImGui::End();

    static float spinPhase = 0.0f;
    if (!state.paused) {
        if (state.animate) state.animPhase += static_cast<float>(dt) * state.animSpeed;
        if (state.autoRotate) spinPhase += static_cast<float>(dt) * state.animSpeed;
    }

    float animX = 0.0f, animY = 0.0f;
    if (state.animate) {
        switch (state.trajectoryType) {
        case 0: // Circle
            animX = state.animRadius * std::cos(state.animPhase);
            animY = state.animRadius * std::sin(state.animPhase);
            break;
        case 1: // Ellipse
            animX = state.animRadius * std::cos(state.animPhase);
            animY = state.animRadius * state.animParam * std::sin(state.animPhase);
            break;
        case 2: // Lissajous
            animX = state.animRadius * std::sin(state.animParam * state.animPhase);
            animY = state.animRadius * std::sin((state.animParam + 1.0f) * state.animPhase);
            break;
        case 3: // Spiral
        {
            float r = state.animRadius * state.animParam * 0.5f * (1.0f - std::cos(state.animPhase / 4.0f));
            animX = r * std::cos(state.animPhase);
            animY = r * std::sin(state.animPhase);
            break;
        }
        }
    }

    float autoRotX = spinPhase * 0.5f;
    float autoRotY = spinPhase * 0.7f;
    float autoRotZ = spinPhase * 0.3f;

    Mat4 model = identity();
    model = multiply(model, translation(state.position[0] + animX, state.position[1] + animY, state.position[2]));
    model = multiply(model, rotationY(state.rotation[1] * PI / 180.0f + autoRotY));
    model = multiply(model, rotationX(state.rotation[0] * PI / 180.0f + autoRotX));
    model = multiply(model, rotationZ(state.rotation[2] * PI / 180.0f + autoRotZ));
    model = multiply(model, scaling(state.scale[0], state.scale[1], state.scale[2]));

    Mat4 view = translation(0.0f, 0.0f, -3.5f);

    auto& ctx = graphics::internal::context;
    float aspect = static_cast<float>(ctx.swapchain_extent.width) / static_cast<float>(ctx.swapchain_extent.height);

    Mat4 proj;
    if (state.projection == 0) {
        proj = perspective(45.0f * PI / 180.0f, aspect, 0.1f, 100.0f);
    } else {
        float h = 2.5f;
        float w = h * aspect;
        proj = ortho(-w, w, -h, h, 0.1f, 100.0f);
    }

    Mat4 mvp = multiply(proj, multiply(view, model));

    pendingUniforms.mvp = mvp;
    pendingUniforms.model = model;
    pendingUniforms.userColor[0] = state.userColor[0];
    pendingUniforms.userColor[1] = state.userColor[1];
    pendingUniforms.userColor[2] = state.userColor[2];
    pendingUniforms.userColor[3] = 1.0f;
}

void render(const graphics::internal::FrameData& fd) {
    if (fd.command_buffer == VK_NULL_HANDLE) return;

    auto& ctx = graphics::internal::context;
    VkCommandBuffer cmd = fd.command_buffer;

    std::memcpy(gpu.uboMapped, &pendingUniforms, sizeof(UniformData));
    vmaFlushAllocation(ctx.allocator, gpu.uboAlloc, 0, sizeof(UniformData));

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

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.pipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(ctx.swapchain_extent.width);
    viewport.height = static_cast<float>(ctx.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &gpu.vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, gpu.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu.pipelineLayout, 0, 1, &gpu.descriptorSet, 0, nullptr);
    vkCmdDrawIndexed(cmd, gpu.indexCount, 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application