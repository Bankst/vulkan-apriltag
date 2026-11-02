#ifndef SHARED_THRESHOLD_H
#define SHARED_THRESHOLD_H

#ifndef __SLANG__
#include <cstdint>
#endif

struct ThresholdPush {
    uint32_t width;
    uint32_t height;
    // width * height bytes of a grayscale image
    uint64_t grayscale;
    // (width/4) * (height/4) * 4 bytes
    // min lo and max hi in 3x3 tile window
    uint64_t filtered;
};

#endif
