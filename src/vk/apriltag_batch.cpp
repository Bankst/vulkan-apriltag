// Batch runner for the Vulkan AprilTag pipeline, for corpus benchmarking.
//
// Reads a manifest (one image path per line) once, initialises the Vulkan
// device once, caches one pipeline per distinct (width,height), and for each
// image runs `warmup + iterations` frames, reporting median per-frame GPU time
// and the decoded detections (ids + corners) as one JSON object per line on
// stdout. Progress goes to stderr. This mirrors the timing done by
// `apriltag_main.cpp` but amortises device/pipeline setup across the corpus so
// thousands of frames are practical.
//
// Corners are emitted in AprilTag's own convention (pixel centres at +0.5); the
// scorer applies the -0.5 shift to match the benchmark's pixel-centre truth.

#include "quad_decoding.h"
#include "vk/apriltag.h"

#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char *const STAGES[] = {"threshold", "segmentation", "boundaries",
                              "cluster_grouping", "fit_quad"};

double median_of(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double min_of(const std::vector<double> &v) {
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

void usage(const char *argv0) {
    std::cerr << "usage: " << argv0
              << " <manifest.txt> [--device <index|name>] "
                 "[--iterations <n>] [--warmup <n>] [--decode-threads <n>]\n"
                 "  manifest: one image path per line; '#' comments ignored\n"
                 "  emits one JSON object per image on stdout\n";
}

} // namespace

int main(int argc, char **argv) {
    support::ContextOptions options;
    int iterations = 10;
    int warmup = 1;
    unsigned decode_threads = 1;
    std::string manifest_path;

    for (int i = 1; i < argc; i++) {
        const bool has_value = i + 1 < argc;
        if (std::strcmp(argv[i], "--device") == 0 && has_value) {
            options.device = argv[++i];
        } else if (std::strcmp(argv[i], "--iterations") == 0 && has_value) {
            iterations = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--warmup") == 0 && has_value) {
            warmup = std::max(0, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--decode-threads") == 0 && has_value) {
            decode_threads = unsigned(std::max(1, std::atoi(argv[++i])));
        } else if (argv[i][0] != '-' && manifest_path.empty()) {
            manifest_path = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (manifest_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    std::vector<std::string> paths;
    {
        std::ifstream in(manifest_path);
        if (!in) {
            std::cerr << "cannot open manifest " << manifest_path << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(in, line)) {
            // Trim trailing whitespace/CR.
            while (!line.empty() &&
                   (line.back() == '\r' || line.back() == '\n' ||
                    line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            if (line.empty() || line[0] == '#') {
                continue;
            }
            paths.push_back(line);
        }
    }
    std::cerr << "manifest: " << paths.size() << " images\n";

    try {
        support::Context ctx(options);
        std::cerr << "device: " << ctx.report.name
                  << "  decode threads: " << decode_threads << "\n";
        QuadDecoder decoder(decode_threads);

        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        // Keep a single pipeline and rebuild it only when the resolution
        // changes. The corpus is grouped by resolution, so this rebuilds just a
        // few times per group, while never holding more than one resolution's
        // device buffers at once (a 2 GB GPU cannot hold all three).
        std::unique_ptr<avk::AprilTagPipeline> pipeline_ptr;
        uint32_t cur_w = 0, cur_h = 0;

        std::cout << std::fixed;
        size_t done = 0;
        for (const std::string &path : paths) {
            int w = 0, h = 0, comp = 0;
            stbi_uc *data = stbi_load(path.c_str(), &w, &h, &comp, STBI_grey);
            if (!data) {
                std::cout << "{\"path\":\"" << path
                          << "\",\"error\":\"load_failed\"}\n";
                std::cerr << "warning: cannot load " << path << ": "
                          << stbi_failure_reason() << "\n";
                continue;
            }
            const std::vector<uint8_t> image(data, data + size_t(w) * h);
            stbi_image_free(data);
            const uint32_t width = uint32_t(w);
            const uint32_t height = uint32_t(h);

            if (!pipeline_ptr || width != cur_w || height != cur_h) {
                // Free the old pipeline's device memory before allocating the
                // new one, so peak VRAM is one resolution, not the sum.
                pipeline_ptr.reset();
                pipeline_ptr = std::make_unique<avk::AprilTagPipeline>(
                    ctx, width, height);
                cur_w = width;
                cur_h = height;
            }
            avk::AprilTagPipeline &pipeline = *pipeline_ptr;
            support::GpuTimer timer(ctx);

            std::vector<double> frame_ms, gpu_total_ms, decode_ms, total_ms;
            std::vector<Detection> detections;
            size_t quad_count = 0;
            bool dropped_any = false;

            for (int rep = 0; rep < warmup + iterations; rep++) {
                const auto t0 = Clock::now();
                pipeline.upload(image.data());
                ctx.submit_and_wait([&](vk::CommandBuffer cmd) {
                    pipeline.record(cmd, &timer);
                });
                const auto t2 = Clock::now();
                const auto corners = pipeline.corners();
                detections = decoder.decode(corners.data(), corners.size(),
                                            width, height, image.data());
                quad_count = corners.size();
                if (pipeline.dropped_points() > 0) {
                    dropped_any = true;
                }
                const auto t3 = Clock::now();
                if (rep < warmup) {
                    continue;
                }
                double gpu_total = 0;
                const auto marks = timer.read();
                for (const auto &[label, t] : marks) {
                    for (const char *stage : STAGES) {
                        if (label.rfind(stage, 0) == 0) {
                            gpu_total += t;
                            break;
                        }
                    }
                }
                frame_ms.push_back(ms(t0, t2));
                decode_ms.push_back(ms(t2, t3));
                total_ms.push_back(ms(t0, t3));
                gpu_total_ms.push_back(gpu_total);
            }

            std::ostringstream os;
            os << std::fixed << std::setprecision(5);
            os << "{\"path\":\"" << path << "\",\"w\":" << width
               << ",\"h\":" << height << ",\"quads\":" << quad_count
               << ",\"dropped\":" << (dropped_any ? "true" : "false")
               << ",\"frame_ms\":" << median_of(frame_ms)
               << ",\"frame_min_ms\":" << min_of(frame_ms)
               << ",\"gpu_total_ms\":" << median_of(gpu_total_ms)
               << ",\"decode_ms\":" << median_of(decode_ms)
               << ",\"total_ms\":" << median_of(total_ms)
               << ",\"total_min_ms\":" << min_of(total_ms) << ",\"dets\":[";
            os << std::setprecision(4);
            for (size_t d = 0; d < detections.size(); d++) {
                const Detection &det = detections[d];
                if (d) {
                    os << ",";
                }
                os << "[" << det.id << ",[";
                for (int k = 0; k < 4; k++) {
                    if (k) {
                        os << ",";
                    }
                    os << "[" << det.corners[k][0] << "," << det.corners[k][1]
                       << "]";
                }
                os << "]]";
            }
            os << "]}";
            std::cout << os.str() << "\n";

            if (++done % 100 == 0) {
                std::cerr << "  " << done << "/" << paths.size() << "\r"
                          << std::flush;
            }
        }
        std::cerr << "\ndone: " << done << " images\n";
        if (ctx.validation_errors) {
            std::cerr << "validation: " << ctx.validation_errors << " errors\n";
            return 1;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
