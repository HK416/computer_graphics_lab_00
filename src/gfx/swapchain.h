#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

struct SDL_Window;

namespace gfx {

struct Context;

struct Swapchain {
    Swapchain(Context& context, SDL_Window* window, bool vsync);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // 윈도우 크기 변경이나 표시 모드 전환 시 기존 스왑체인을 재사용해 다시 만든다.
    void recreate(bool vsync);

    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

private:
    void destroyImageViews();

    Context& context;
    SDL_Window* window = nullptr;
};

} // namespace gfx
