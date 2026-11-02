#include "vk/threshold.h"

#include "shaders/threshold.slang.h"

namespace avk {

static constexpr uint32_t WG_X = 16;
static constexpr uint32_t WG_Y = 8;
static constexpr uint32_t WG_SIZE[] = {WG_X, WG_Y};

static uint32_t groups(uint32_t n, uint32_t size) {
    return (n + size - 1) / size;
}

Threshold::Threshold(const support::Context &ctx)
    : minmax_filter(support::make_pipeline(
          ctx, threshold_slang, sizeof(ThresholdPush), WG_SIZE,
          "minmax_filter")) {}

void Threshold::record(vk::CommandBuffer cmd, const ThresholdPush &push,
                       support::GpuTimer *timer) const {
    if (push.width % 4 != 0 || push.height % 4 != 0) {
        throw std::runtime_error(
            "threshold: image size " + std::to_string(push.width) + "x" +
            std::to_string(push.height) + " is not a multiple of 4");
    }
    const uint32_t tiles_w = push.width / 4;
    const uint32_t tiles_h = push.height / 4;
    const auto compute = vk::PipelineBindPoint::eCompute;
    const auto stage = vk::ShaderStageFlagBits::eCompute;

    cmd.bindPipeline(compute, *minmax_filter.pipeline);
    cmd.pushConstants<ThresholdPush>(*minmax_filter.layout, stage, 0, push);
    cmd.dispatch(groups(tiles_w, WG_X), groups(tiles_h, WG_Y), 1);
    if (timer) {
        timer->mark(cmd, "threshold");
    }
}

} // namespace avk
