// CPU AprilTag (tag36h11) detect+decode benchmark, for comparison against the
// Vulkan pipeline. Reports median/min ms over N iterations at a given
// quad_decimate, plus detected ids. Decimation is the stock CPU speed lever.
#include "apriltag.h"
#include "tag36h11.h"
#include "common/image_u8.h"
#include "common/zarray.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <image.png> [decimate=1.0] [iters=20] "
                        "[warmup=5] [threads=4]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    float decimate = argc > 2 ? atof(argv[2]) : 1.0f;
    int iters = argc > 3 ? atoi(argv[3]) : 20;
    int warmup = argc > 4 ? atoi(argv[4]) : 5;
    int threads = argc > 5 ? atoi(argv[5]) : 4;

    int w, h, comp;
    stbi_uc *data = stbi_load(path, &w, &h, &comp, 1);
    if (!data) { fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }

    image_u8_t *im = image_u8_create((unsigned)w, (unsigned)h);
    for (int y = 0; y < h; y++)
        memcpy(im->buf + (size_t)y * im->stride, data + (size_t)y * w, w);
    stbi_image_free(data);

    apriltag_family_t *fam = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, fam);
    td->quad_decimate = decimate;
    td->quad_sigma = 0.0f;
    td->nthreads = threads;
    td->refine_edges = true;
    td->decode_sharpening = 0.25;
    td->debug = false;

    double *t = malloc(sizeof(double) * iters);
    zarray_t *last_dets = NULL;
    for (int it = 0; it < warmup + iters; it++) {
        double a = now_ms();
        zarray_t *dets = apriltag_detector_detect(td, im);
        double b = now_ms();
        if (it >= warmup) t[it - warmup] = b - a;
        if (last_dets) apriltag_detections_destroy(last_dets);
        last_dets = dets;
    }
    qsort(t, iters, sizeof(double), cmp_d);
    double med = t[iters / 2], mn = t[0];

    int ndet = last_dets ? zarray_size(last_dets) : 0;
    printf("decimate=%.1f threads=%d  %dx%d  -> %d tag(s)\n",
           decimate, threads, w, h, ndet);
    for (int i = 0; i < ndet; i++) {
        apriltag_detection_t *d;
        zarray_get(last_dets, i, &d);
        printf("  id %d (hamming %d, margin %.1f) centre (%.2f, %.2f)"
               " corners (%.2f, %.2f) (%.2f, %.2f) (%.2f, %.2f) (%.2f, %.2f)\n",
               d->id, d->hamming, d->decision_margin,
               d->c[0], d->c[1],
               d->p[0][0], d->p[0][1], d->p[1][0], d->p[1][1],
               d->p[2][0], d->p[2][1], d->p[3][0], d->p[3][1]);
    }
    if (last_dets) apriltag_detections_destroy(last_dets);
    printf("  detect+decode median/min: %.3f / %.3f ms\n", med, mn);

    free(t);
    apriltag_detector_destroy(td);
    tag36h11_destroy(fam);
    return 0;
}
