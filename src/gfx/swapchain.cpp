#include "gfx/swapchain.h"

#include <algorithm>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include "core/error.h"
#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

VkSurfaceFormatKHR chooseSurfaceFormat(VkPhysicalDevice device, VkSurfaceKHR surface) {
    uint32_t count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, formats.data()));
    if (formats.empty()) {
        core::fatal("서피스가 지원하는 포맷이 없습니다");
    }

    for (const VkSurfaceFormatKHR& format : formats) {
        bool preferred = format.format == VK_FORMAT_B8G8R8A8_SRGB || format.format == VK_FORMAT_R8G8B8A8_SRGB;
        if (preferred && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats.front();
}

VkPresentModeKHR choosePresentMode(VkPhysicalDevice device, VkSurfaceKHR surface, bool vsync) {
    if (vsync) {
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    uint32_t count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr));
    std::vector<VkPresentModeKHR> modes(count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, modes.data()));

    auto supports = [&modes](VkPresentModeKHR mode) { return std::ranges::find(modes, mode) != modes.end(); };
    if (supports(VK_PRESENT_MODE_MAILBOX_KHR)) {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }
    if (supports(VK_PRESENT_MODE_IMMEDIATE_KHR)) {
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, SDL_Window* window) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    // 고밀도 디스플레이에서는 논리 크기가 아니라 픽셀 크기를 써야 한다.
    int width = 0;
    int height = 0;
    SDL_GetWindowSizeInPixels(window, &width, &height);
    VkExtent2D extent{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return extent;
}

} // namespace

Swapchain::Swapchain(Context& context, SDL_Window* window, bool vsync) : context(context), window(window) {
    recreate(vsync);
}

Swapchain::~Swapchain() {
    destroyImageViews();
    vkDestroySwapchainKHR(context.device, handle, nullptr);
}

void Swapchain::destroyImageViews() {
    for (VkImageView view : imageViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    imageViews.clear();
}

void Swapchain::recreate(bool vsync) {
    VkSurfaceCapabilitiesKHR capabilities{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.physicalDevice, context.surface, &capabilities));

    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(context.physicalDevice, context.surface);
    format = surfaceFormat.format;
    colorSpace = surfaceFormat.colorSpace;
    presentMode = choosePresentMode(context.physicalDevice, context.surface, vsync);
    extent = chooseExtent(capabilities, window);

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }

    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = context.surface;
    info.minImageCount = imageCount;
    info.imageFormat = format;
    info.imageColorSpace = colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = capabilities.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = presentMode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = handle;

    VkSwapchainKHR created = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(context.device, &info, nullptr, &created));

    destroyImageViews();
    vkDestroySwapchainKHR(context.device, handle, nullptr);
    handle = created;

    uint32_t count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(context.device, handle, &count, nullptr));
    images.resize(count);
    VK_CHECK(vkGetSwapchainImagesKHR(context.device, handle, &count, images.data()));

    imageViews.reserve(count);
    for (VkImage image : images) {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(context.device, &viewInfo, nullptr, &view));
        imageViews.push_back(view);
    }

    spdlog::info("스왑체인 {}x{}, 이미지 {}장, 표시 모드 {}",
                 extent.width,
                 extent.height,
                 count,
                 presentMode == VK_PRESENT_MODE_FIFO_KHR ? "FIFO" : "MAILBOX/IMMEDIATE");
}

} // namespace gfx
