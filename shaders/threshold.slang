// Apriltag style adaptive thresholding.
#include "shared/threshold.h"

[vk::push_constant]
ConstantBuffer<ThresholdPush> pc;

[vk::constant_id(0)]
const uint32_t WG_X = 1;
[vk::constant_id(1)]
const uint32_t WG_Y = 1;

// Fused minmax + filter: each thread computes one tile's min/max, then uses
// shared memory to apply the 3x3 neighborhood filter without a second dispatch.
static const uint32_t HALO = 1;
static const uint32_t SHARED_W = WG_X + 2 * HALO;
static const uint32_t SHARED_H = WG_Y + 2 * HALO;
groupshared uint32_t shared_extrema[SHARED_H * SHARED_W];

uint32_t compute_tile_extrema(uint32_t tx, uint32_t ty) {
    uint32_t *grayscale = (uint32_t *)pc.grayscale;
    const uint32_t tiles_w = pc.width / 4;

    if (tx >= tiles_w || ty >= pc.height / 4) {
        return 0xff;
    }

    uint32_t lo = 255;
    uint32_t hi = 0;
    for (uint32_t row = 0; row < 4; row++) {
        const uint32_t word = grayscale[(ty * 4 + row) * tiles_w + tx];
        for (uint32_t b = 0; b < 4; b++) {
            const uint32_t value = bitfieldExtract(word, int32_t(b * 8), 8);
            lo = min(lo, value);
            hi = max(hi, value);
        }
    }
    return lo | (hi << 8);
}

[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void minmax_filter(uint3 group_id: SV_GroupID, uint3 local_id: SV_GroupThreadID) {
    uint32_t *filtered = (uint32_t *)pc.filtered;

    const int32_t tiles_w = int32_t(pc.width / 4);
    const int32_t tiles_h = int32_t(pc.height / 4);

    // The tile coordinate this thread is responsible for writing.
    const int32_t tx = int32_t(group_id.x * WG_X + local_id.x);
    const int32_t ty = int32_t(group_id.y * WG_Y + local_id.y);

    // Base tile of the shared memory region (top-left corner including halo).
    const int32_t base_x = int32_t(group_id.x * WG_X) - int32_t(HALO);
    const int32_t base_y = int32_t(group_id.y * WG_Y) - int32_t(HALO);

    // Load shared memory: each thread loads its own tile, plus halo tiles are
    // loaded by threads near the edges. Total shared region is (WG_X+2)*(WG_Y+2).
    const uint32_t thread_count = WG_X * WG_Y;
    const uint32_t total_shared = SHARED_W * SHARED_H;
    const uint32_t linear = local_id.y * WG_X + local_id.x;
    for (uint32_t i = linear; i < total_shared; i += thread_count) {
        const int32_t sx = base_x + int32_t(i % SHARED_W);
        const int32_t sy = base_y + int32_t(i / SHARED_W);
        shared_extrema[i] = compute_tile_extrema(
            uint32_t(clamp(sx, 0, tiles_w - 1)),
            uint32_t(clamp(sy, 0, tiles_h - 1)));
    }
    GroupMemoryBarrierWithGroupSync();

    if (tx >= tiles_w || ty >= tiles_h) {
        return;
    }

    // 3x3 filter over shared memory.
    const uint32_t cx = local_id.x + HALO;
    const uint32_t cy = local_id.y + HALO;
    uint32_t lo = 255;
    uint32_t hi = 0;
    for (int32_t dy = -1; dy <= 1; dy++) {
        for (int32_t dx = -1; dx <= 1; dx++) {
            const uint32_t e = shared_extrema[(cy + dy) * SHARED_W + (cx + dx)];
            lo = min(lo, e & 0xff);
            hi = max(hi, (e >> 8) & 0xff);
        }
    }
    filtered[ty * tiles_w + tx] = lo | (hi << 8);
}

