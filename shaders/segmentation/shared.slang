#include "shared/segmentation.h"

// BKE information bits, stored in slot 2 of each block.
static const uint32_t TOP_LEFT_255 = 1u << 0;
static const uint32_t TOP_RIGHT_255 = 1u << 1;
static const uint32_t BOTTOM_LEFT_255 = 1u << 2;
static const uint32_t BOTTOM_RIGHT_255 = 1u << 3;
static const uint32_t BITMASK_POS_255 = 0xfu;

// This is unused in the original paper, but has been placed
// here to allow some nicer code structure.
static const uint32_t MUST_UNION_P_255 = 1u << 4;
static const uint32_t MUST_UNION_Q_255 = 1u << 5;
static const uint32_t MUST_UNION_R_255 = 1u << 6;
static const uint32_t MUST_UNION_S_255 = 1u << 7;

// Our own unique bits because we have to duplicate them.
static const uint32_t TOP_LEFT_0 = 1u << 8;
static const uint32_t TOP_RIGHT_0 = 1u << 9;
static const uint32_t BOTTOM_LEFT_0 = 1u << 10;
static const uint32_t BOTTOM_RIGHT_0 = 1u << 11;
static const uint32_t BITMASK_POS_0 = 0xfu << 8;

static const uint32_t SPLIT_0 = 1u << 12;
static const uint32_t MUST_UNION_S_FIRST_0 = 1u << 13;
static const uint32_t MUST_UNION_S_SECOND_0 = 1u << 14;

static const uint32_t THRESHOLD_MIN_CONTRAST = 5;

uint32_t pixel(uint64_t grayscale_addr, uint64_t filtered_addr,
               uint32_t width, uint32_t index) {
    uint32_t *grayscale = (uint32_t *)grayscale_addr;
    uint32_t *filtered = (uint32_t *)filtered_addr;

    const uint32_t gray = bitfieldExtract(
        grayscale[index >> 2], int32_t((index & 3) * 8), 8);

    const uint32_t px = index % width;
    const uint32_t py = index / width;
    const uint32_t f = filtered[(py / 4) * (width / 4) + (px / 4)];
    const uint32_t lo = f & 0xff;
    const uint32_t hi = (f >> 8) & 0xff;

    if (hi - lo < THRESHOLD_MIN_CONTRAST) {
        return 127;
    }
    const uint32_t mid = lo + (hi - lo) / 2;
    return gray > mid ? 255u : 0u;
}
