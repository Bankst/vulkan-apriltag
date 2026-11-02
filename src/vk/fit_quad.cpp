#include "vk/fit_quad.h"

#include "shaders/fit_quad/maxima.slang.h"
#include "shaders/fit_quad/prepare_indirect.slang.h"
#include "shaders/fit_quad/scan.slang.h"
#include "shaders/fit_quad/search.slang.h"
#include "shaders/fit_quad/sort.slang.h"

#include <algorithm>

namespace avk {
constexpr uint32_t GROUPS = 256;
constexpr uint32_t SORT_GROUPS = 512;

constexpr uint32_t SEARCH_WG_SIZE[] = {128};

// Below this pixel count, direct dispatch with fixed group counts is faster
// than the prepare_indirect dispatch + compute→indirect barrier overhead.
constexpr uint32_t INDIRECT_PIXEL_THRESHOLD = 500'000;

static uint32_t pick_wg_size(const support::Context &ctx) {
    if (!ctx.caps.int16_storage) {
        throw std::runtime_error("quad fitting needs storageBuffer16BitAccess "
                                 "and shaderInt16 for its 16-bit cluster "
                                 "indexes, which " +
                                 ctx.report.name + " lacks");
    }
    return ctx.caps.max_workgroup_invocations >= 256 ? 256 : 128;
}

static void mark(vk::CommandBuffer cmd, support::GpuTimer *timer,
                 const char *label) {
    if (timer) {
        timer->mark(cmd, label);
    }
}

static void dispatch(vk::CommandBuffer cmd, const support::Pipeline &pipeline,
                     const FitQuadPush &push, uint32_t groups,
                     support::GpuTimer *timer, const char *label = nullptr,
                     bool last = false) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline.pipeline);
    cmd.pushConstants<FitQuadPush>(*pipeline.layout,
                                   vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch(groups, 1, 1);
    if (label) {
        mark(cmd, timer, label);
    }
    if (!last) {
        support::compute_to_compute_barrier(cmd);
    }
}

static void dispatch_indirect(vk::CommandBuffer cmd,
                              const support::Pipeline &pipeline,
                              const FitQuadPush &push, vk::Buffer buffer,
                              vk::DeviceSize offset,
                              support::GpuTimer *timer,
                              const char *label = nullptr,
                              bool last = false) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline.pipeline);
    cmd.pushConstants<FitQuadPush>(*pipeline.layout,
                                   vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatchIndirect(buffer, offset);
    if (label) {
        mark(cmd, timer, label);
    }
    if (!last) {
        support::compute_to_compute_barrier(cmd);
    }
}

FitQuad::FitQuad(const support::Context &ctx) : wg_size(pick_wg_size(ctx)) {
    const uint32_t spec[] = {wg_size};
    const auto make = [&](std::span<const uint32_t> spirv, const char *entry,
                          std::span<const uint32_t> spec_constants) {
        return support::make_pipeline(ctx, spirv, sizeof(FitQuadPush),
                                      spec_constants, entry);
    };
    prepare_indirect = make(fit_quad_prepare_indirect_slang, "main", spec);
    units = make(fit_quad_sort_slang, "units", spec);
    sort = make(fit_quad_sort_slang, "sort", spec);
    merge = make(fit_quad_sort_slang, "merge", spec);
    gather = make(fit_quad_sort_slang, "gather", spec);
    scan_reduce = make(fit_quad_scan_slang, "scan_reduce", spec);
    scan_fused = make(fit_quad_scan_slang, "scan_fused", spec);
    window_errors = make(fit_quad_maxima_slang, "window_errors", spec);
    filter = make(fit_quad_maxima_slang, "filter", spec);
    maxima = make(fit_quad_maxima_slang, "maxima", spec);
    search = make(fit_quad_search_slang, "search", SEARCH_WG_SIZE);
}

uint32_t FitQuad::merge_rounds(uint32_t width, uint32_t height) {
    const uint32_t largest =
        std::min(MAX_POINTS_PER_PIXEL * width * height, 4 * (width + height));
    uint32_t rounds = 0;
    while ((SORT_CHUNK << rounds) < largest) {
        rounds++;
    }
    return rounds;
}

void FitQuad::record(vk::CommandBuffer cmd, FitQuadPush push,
                     vk::Buffer arena_buffer, vk::DeviceSize arena_offset,
                     support::GpuTimer *timer) const {
    push.round = 0;
    const bool use_indirect =
        uint64_t(push.width) * push.height >= INDIRECT_PIXEL_THRESHOLD;

    if (use_indirect) {
        dispatch(cmd, prepare_indirect, push, 1, timer, "fit_quad_prepare");
        support::barrier(cmd, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderStorageWrite,
                         vk::PipelineStageFlagBits2::eDrawIndirect,
                         vk::AccessFlagBits2::eIndirectCommandRead);
    }

    const vk::DeviceSize indirect_base =
        arena_offset +
        FIT_QUAD_INDIRECT_OFFSET(MAX_POINTS_PER_PIXEL * push.width *
                                 push.height);

    auto ioff = [&](uint32_t index) -> vk::DeviceSize {
        return indirect_base + vk::DeviceSize(index) * 12;
    };

    record_sort(cmd, push, use_indirect, arena_buffer, ioff, timer);
    record_scan(cmd, push, use_indirect, arena_buffer, ioff, timer);
    record_maxima(cmd, push, use_indirect, arena_buffer, ioff, timer);
    record_search(cmd, push, use_indirect, arena_buffer, ioff, timer);
}

void FitQuad::record_sort(vk::CommandBuffer cmd, FitQuadPush push,
                          bool indirect, vk::Buffer buffer,
                          const IndirectFn &ioff,
                          support::GpuTimer *timer) const {
    if (indirect) {
        dispatch_indirect(cmd, units, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_UNITS), timer,
                          "fit_quad_units");
        dispatch_indirect(cmd, sort, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_SORT), timer,
                          "fit_quad_sort");
        const uint32_t rounds = merge_rounds(push.width, push.height);
        for (uint32_t round = 0; round < rounds; round++) {
            push.round = round;
            dispatch_indirect(cmd, merge, push, buffer,
                              ioff(FIT_QUAD_INDIRECT_SORT), timer);
        }
        push.round = 0;
        mark(cmd, timer, "fit_quad_merge");
        dispatch_indirect(cmd, gather, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_SORT), timer,
                          "fit_quad_gather");
    } else {
        dispatch(cmd, units, push, GROUPS, timer, "fit_quad_units");
        dispatch(cmd, sort, push, SORT_GROUPS, timer, "fit_quad_sort");
        const uint32_t rounds = merge_rounds(push.width, push.height);
        for (uint32_t round = 0; round < rounds; round++) {
            push.round = round;
            dispatch(cmd, merge, push, SORT_GROUPS, timer);
        }
        push.round = 0;
        mark(cmd, timer, "fit_quad_merge");
        dispatch(cmd, gather, push, SORT_GROUPS, timer, "fit_quad_gather");
    }
}

void FitQuad::record_scan(vk::CommandBuffer cmd, const FitQuadPush &push,
                          bool indirect, vk::Buffer buffer,
                          const IndirectFn &ioff,
                          support::GpuTimer *timer) const {
    if (indirect) {
        dispatch_indirect(cmd, scan_reduce, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_SCAN_REDUCE), timer,
                          "fit_quad_scan_reduce");
        dispatch_indirect(cmd, scan_fused, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_SCAN_FUSED), timer,
                          "fit_quad_scan");
    } else {
        dispatch(cmd, scan_reduce, push, GROUPS, timer,
                 "fit_quad_scan_reduce");
        dispatch(cmd, scan_fused, push, GROUPS, timer, "fit_quad_scan");
    }
}

void FitQuad::record_maxima(vk::CommandBuffer cmd, const FitQuadPush &push,
                            bool indirect, vk::Buffer buffer,
                            const IndirectFn &ioff,
                            support::GpuTimer *timer) const {
    if (indirect) {
        dispatch_indirect(cmd, window_errors, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_WINDOW_ERRORS), timer,
                          "fit_quad_window_errors");
        dispatch_indirect(cmd, filter, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_FILTER), timer,
                          "fit_quad_filter");
        dispatch_indirect(cmd, maxima, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_MAXIMA), timer,
                          "fit_quad_maxima");
    } else {
        dispatch(cmd, window_errors, push, GROUPS, timer,
                 "fit_quad_window_errors");
        dispatch(cmd, filter, push, GROUPS, timer, "fit_quad_filter");
        dispatch(cmd, maxima, push, GROUPS, timer, "fit_quad_maxima");
    }
}

void FitQuad::record_search(vk::CommandBuffer cmd, const FitQuadPush &push,
                            bool indirect, vk::Buffer buffer,
                            const IndirectFn &ioff,
                            support::GpuTimer *timer) const {
    if (indirect) {
        dispatch_indirect(cmd, search, push, buffer,
                          ioff(FIT_QUAD_INDIRECT_SEARCH), timer,
                          "fit_quad_search", true);
    } else {
        dispatch(cmd, search, push, GROUPS, timer, "fit_quad_search", true);
    }
}

} // namespace avk
