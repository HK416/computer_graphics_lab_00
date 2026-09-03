#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

#include <spdlog/spdlog.h>

#include "asset/model.h"

// KTX2 컨테이너에 미리 압축해 둔 BC 텍스처를 읽는다. 초압축(Basis/zstd)은 다루지 않고 블록 데이터가
// 그대로 들어 있는 파일만 받는다. 트랜스코더 의존성 없이 헤더와 단계 표만 읽으면 되기 때문이다.
// `ktx create --format BC7_SRGB_BLOCK --generate-mipmap` 같은 도구가 이 형식을 낸다.
namespace asset {
namespace {

constexpr std::array<uint8_t, 12> KTX2_IDENTIFIER{
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

// 헤더 뒤에 오는 단계 표 항목. 파일 안의 위치와 길이다.
struct LevelIndex {
    uint64_t byteOffset;
    uint64_t byteLength;
    uint64_t uncompressedByteLength;
};
static_assert(sizeof(LevelIndex) == 24, "단계 표 항목은 파일 배치 그대로 24 바이트여야 한다");

template <typename T> T readAt(const std::vector<uint8_t>& bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

// VkFormat 값. 색 공간은 재질 슬롯이 정하므로 sRGB 와 UNORM 을 같은 포맷으로 본다.
bool formatFromVulkan(uint32_t vkFormat, TextureFormat& format) {
    switch (vkFormat) {
    case 37: // VK_FORMAT_R8G8B8A8_UNORM
    case 43: // VK_FORMAT_R8G8B8A8_SRGB
        format = TextureFormat::RGBA8;
        return true;
    case 131: // VK_FORMAT_BC1_RGB_UNORM_BLOCK
    case 132: // VK_FORMAT_BC1_RGB_SRGB_BLOCK
    case 133: // VK_FORMAT_BC1_RGBA_UNORM_BLOCK
    case 134: // VK_FORMAT_BC1_RGBA_SRGB_BLOCK
        format = TextureFormat::BC1;
        return true;
    case 137: // VK_FORMAT_BC3_UNORM_BLOCK
    case 138: // VK_FORMAT_BC3_SRGB_BLOCK
        format = TextureFormat::BC3;
        return true;
    case 139: // VK_FORMAT_BC4_UNORM_BLOCK
        format = TextureFormat::BC4;
        return true;
    case 141: // VK_FORMAT_BC5_UNORM_BLOCK
        format = TextureFormat::BC5;
        return true;
    case 145: // VK_FORMAT_BC7_UNORM_BLOCK
    case 146: // VK_FORMAT_BC7_SRGB_BLOCK
        format = TextureFormat::BC7;
        return true;
    default:
        return false;
    }
}

} // namespace

bool isBlockCompressed(TextureFormat format) {
    return format != TextureFormat::RGBA8;
}

size_t textureLevelBytes(TextureFormat format, uint32_t width, uint32_t height) {
    if (format == TextureFormat::RGBA8) {
        return static_cast<size_t>(width) * height * 4;
    }
    // 블록 압축은 4x4 블록 단위이고 BC1/BC4 는 블록당 8 바이트, 나머지는 16 바이트다.
    size_t blockBytes = (format == TextureFormat::BC1 || format == TextureFormat::BC4) ? 8 : 16;
    size_t blocksX = (static_cast<size_t>(width) + 3) / 4;
    size_t blocksY = (static_cast<size_t>(height) + 3) / 4;
    return blocksX * blocksY * blockBytes;
}

bool isKtx2(const std::vector<uint8_t>& bytes) {
    return bytes.size() >= KTX2_IDENTIFIER.size() &&
           std::equal(KTX2_IDENTIFIER.begin(), KTX2_IDENTIFIER.end(), bytes.begin());
}

bool loadKtx2(const std::vector<uint8_t>& bytes, Texture& texture) {
    // 식별자 12 + 헤더 필드 9×4 + 색인 4×4 + 2×8 = 80 바이트.
    constexpr size_t HEADER_SIZE = 80;
    if (!isKtx2(bytes) || bytes.size() < HEADER_SIZE) {
        spdlog::error("KTX2 헤더가 아닙니다: {}", texture.name);
        return false;
    }
    uint32_t vkFormat = readAt<uint32_t>(bytes, 12);
    uint32_t width = readAt<uint32_t>(bytes, 20);
    uint32_t height = readAt<uint32_t>(bytes, 24);
    uint32_t depth = readAt<uint32_t>(bytes, 28);
    uint32_t layerCount = readAt<uint32_t>(bytes, 32);
    uint32_t faceCount = readAt<uint32_t>(bytes, 36);
    uint32_t levelCount = readAt<uint32_t>(bytes, 40);
    uint32_t supercompression = readAt<uint32_t>(bytes, 44);

    TextureFormat format = TextureFormat::RGBA8;
    if (!formatFromVulkan(vkFormat, format)) {
        spdlog::error(
            "지원하지 않는 KTX2 포맷(VkFormat {}): {}. BC1/3/4/5/7 과 RGBA8 만 읽는다", vkFormat, texture.name);
        return false;
    }
    if (supercompression != 0) {
        spdlog::error("초압축된 KTX2(scheme {}) 는 읽지 않습니다: {}. 블록을 그대로 담은 파일이어야 한다",
                      supercompression,
                      texture.name);
        return false;
    }
    if (width == 0 || height == 0 || depth > 1 || layerCount > 1 || faceCount != 1) {
        spdlog::error("2D 한 장이 아닌 KTX2 는 읽지 않습니다: {} ({}x{}x{}, 층 {}, 면 {})",
                      texture.name,
                      width,
                      height,
                      depth,
                      layerCount,
                      faceCount);
        return false;
    }
    // 규격상 0 은 «밉을 만들어 쓰라» 는 뜻이지만 압축 포맷은 밉을 만들 수 없으므로 한 단계로 본다.
    levelCount = std::max(levelCount, 1U);
    uint32_t maxLevels = 1U + static_cast<uint32_t>(std::bit_width(std::max(width, height))) - 1U;
    if (levelCount > maxLevels) {
        spdlog::error("KTX2 밉 단계가 크기보다 많습니다: {} ({} 단계, 최대 {})", texture.name, levelCount, maxLevels);
        return false;
    }
    if (bytes.size() < HEADER_SIZE + static_cast<size_t>(levelCount) * sizeof(LevelIndex)) {
        spdlog::error("KTX2 단계 표가 잘렸습니다: {}", texture.name);
        return false;
    }

    std::vector<uint8_t> pixels;
    for (uint32_t level = 0; level < levelCount; ++level) {
        auto entry = readAt<LevelIndex>(bytes, HEADER_SIZE + static_cast<size_t>(level) * sizeof(LevelIndex));
        uint32_t levelWidth = std::max(width >> level, 1U);
        uint32_t levelHeight = std::max(height >> level, 1U);
        size_t expected = textureLevelBytes(format, levelWidth, levelHeight);
        if (entry.byteLength != expected) {
            spdlog::error("KTX2 밉 {} 크기가 맞지 않습니다: {} ({} 바이트, {} 기대)",
                          level,
                          texture.name,
                          entry.byteLength,
                          expected);
            return false;
        }
        // 덧셈이 넘칠 수 있는 값이라 뺄셈으로 비교한다.
        if (entry.byteOffset > bytes.size() || entry.byteLength > bytes.size() - entry.byteOffset) {
            spdlog::error("KTX2 밉 {} 가 파일을 벗어납니다: {} (파일이 잘렸을 수 있다)", level, texture.name);
            return false;
        }
        pixels.insert(pixels.end(),
                      bytes.begin() + static_cast<std::ptrdiff_t>(entry.byteOffset),
                      bytes.begin() + static_cast<std::ptrdiff_t>(entry.byteOffset + entry.byteLength));
    }

    texture.width = width;
    texture.height = height;
    texture.mipLevels = levelCount;
    texture.format = format;
    texture.pixels = std::move(pixels);
    return true;
}

} // namespace asset
