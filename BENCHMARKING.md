# Build & Benchmarking Handoff

This document lets another agent (or person) reproduce the full build and the
GPU-vs-CPU AprilTag benchmarks on a different system. It is self-contained:
follow it top to bottom.

The pipeline detects **tag36h11** tags. The GPU does threshold → segmentation →
boundaries → cluster grouping → quad fitting; the CPU (stock libapriltag) only
**decodes** the quads the GPU found. "Decimation" (`quad_decimate`) is *not* a
knob in the GPU pipeline — it is the stock **CPU detector**'s speed lever, so
the "with/without decimation" comparison is CPU-side only.

---

## 1. Prerequisites

| Tool | Version used | Notes |
|------|--------------|-------|
| CMake | ≥ 3.22 (tested 3.31) | |
| C/C++ compiler | GCC 14 / Clang 19 | needs C++20 |
| Vulkan SDK (headers + loader) | 1.3+ (tested 1.4.309) | `libvulkan-dev` on Debian |
| A Vulkan 1.3 device | Quadro P620 / llvmpipe | `mesa-vulkan-drivers` gives a CPU (llvmpipe) fallback |
| Ninja | any | or use `-G "Unix Makefiles"` |
| `slangc` (Shader-Slang) | **2026.20** | NOT the `libslang2` apt package; see §2 |
| Internet | — | CMake FetchContent clones `apriltag` at configure time |

Quick check:
```bash
cmake --version; gcc --version | head -1
pkg-config --modversion vulkan      # or: ls /usr/include/vulkan/vulkan.h
```

## 2. Install `slangc`

The build needs the **Shader-Slang** compiler `slangc`. It is usually not
packaged; download the official release (Linux x86_64 shown):

```bash
mkdir -p ~/.slangc && cd ~/.slangc
curl -sL https://github.com/shader-slang/slang/releases/download/v2026.20/slang-2026.20-linux-x86_64.tar.gz \
  | tar xz
~/.slangc/bin/slangc -v         # expect: 2026.20
export PATH="$HOME/.slangc/bin:$PATH"   # add to shell profile to persist
```
The binary's RUNPATH is `$ORIGIN/../lib`, so keep the `bin/` and `lib/` dirs
together. Any recent slangc (≥ 2025.x) should work; if you use a different
version, see §3.

## 3. Known build fix (slangc ≥ 2026.x strictness)

The shaders in this tree already include three fixes required by newer slangc
under the project's `-warnings-as-errors all`. If you build a *fresh checkout of
the original commit* with a recent slangc and hit errors, apply these (all are
behavior-preserving):

- **`shaders/cluster_grouping/scan.slang`** and **`shaders/boundaries/find_points.slang`**:
  add `[require(spvGroupNonUniform)]` above the existing
  `[require(spvGroupNonUniformArithmetic)]` on each compute entry point.
  *(Hard error E36104 "undeclared capability": `subgroup_id()` / `WaveIsFirstLane`
  use the base `GroupNonUniform` capability, which newer slangc no longer infers
  from the Arithmetic/Ballot requires.)*
- **`shaders/boundaries/accumulate.slang`**: change `ClusterAccumulator pending;`
  to `ClusterAccumulator pending = {};`.
  *(False-positive warning E41035: `pending` is only read when
  `pending_cluster != 0`, which guarantees prior assignment. The zero-init
  silences the stricter flow analysis with no behavioral change.)*

With an older slangc these may be unnecessary (it inferred the capability and did
not emit the uninit warning), but the fixes are harmless there.

## 4. Build

```bash
cd <repo>
export PATH="$HOME/.slangc/bin:$PATH"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
Produces `build/avk_devices` and `build/vulktan_apriltag`.

## 5. Smoke test — enumerate devices

```bash
./build/avk_devices
```
Lists every Vulkan device and whether it is `suitable` (needs Vulkan 1.3,
int64/int16, buffer device address, scalar block layout, vulkan memory model,
subgroup ballot+arithmetic, compute queue, timestamps). Exit 0 if any suitable.
Use the index or a name substring with `--device`.

## 6. Download benchmark images

This project uses the [NAPA AprilTag Synthetic Benchmark](https://huggingface.co/datasets/zwatx/napa-apriltag-benchmark)
(v1.0.0) — 4,500 grayscale PNGs with subpixel ground-truth corners, tag36h11.

Three splits: **main** (3,000 images, 20 scenarios), **stress** (600 images,
small tags / heavy noise / dense layouts), **range** (900 images, tag sizes
12–96 px). Resolutions: 640×480, 1280×720, 1920×1080.

Requires `huggingface-cli` (`pip install huggingface_hub`):
```bash
./bench/download_dataset.sh          # downloads "main" split (~default)
./bench/download_dataset.sh stress   # just the stress split
./bench/download_dataset.sh all      # all 4,500 images
```
Images land in `bench/dataset/data/{main,stress,range}/`.

### Build the CPU bench harness

```bash
SRC=build/_deps/apriltag-src
LIB=build/_deps/apriltag-build
RPATH="$PWD/$LIB"
cc -O2 -I"$SRC" -Iinclude bench/cpu_bench.c -L"$LIB" -lapriltag -lm -lpthread -Wl,-rpath,"$RPATH" -o build/cpu_bench
```

## 7. Run the benchmarks

### Automated benchmark (recommended)

`bench/benchmark.py` runs both detectors against the dataset, compares
detections to ground truth (tag IDs + corner positions), and reports precision,
recall, and mean corner error per scenario:

```bash
python3 bench/benchmark.py --split main --iterations 5 --warmup 2
python3 bench/benchmark.py --split stress --gpu-only --summary
python3 bench/benchmark.py --split main --scenario blur --limit 10
python3 bench/benchmark.py --split main --cpu-only --decimate 2.0
```

Key flags: `--gpu-only` / `--cpu-only`, `--scenario <substring>`,
`--limit N`, `--summary`, `--json`, `--decimate`, `--threads`.

### Manual single-image runs

GPU pipeline:
```bash
./build/vulktan_apriltag bench/dataset/data/main/many-tags-1280x720-000.png \
    --device 0 --iterations 50 --warmup 10
```

CPU detector:
```bash
./build/cpu_bench bench/dataset/data/main/many-tags-1280x720-000.png 1.0 50 10 4
```

Other flags: `--device <idx|name>`, `--validation`, `--staging`,
`--decode-threads <n>`, `--dispatches`.

## 8. FRC field images

Real-world FRC camera captures are in `bench/images/frc/`. These have no
machine-readable ground truth but are useful for qualitative testing:
```bash
./build/vulktan_apriltag bench/images/frc/4322_1.png --device 0
```

## 9. Caveats
- The NAPA dataset is synthetic (perspective transforms, noise, blur, clutter,
  but no real optical effects). Real scenes will differ.
- GPU `frame` includes submit/wait synchronization latency (~1 ms); a pipelined
  deployment overlapping upload/compute/decode hides most of it.
- `cpu_bench` uses `quad_sigma=0`, `refine_edges=true`, `decode_sharpening=0.25`,
  tag36h11 — match these if comparing elsewhere.
- The dataset is GPL-3.0. See its LICENSE and NOTICE files for attribution.

---

## 10. Head-to-head vs. NAPA and WPILib (full corpus)

§6–§7 compare the GPU against stock AprilTag. This section adds the two
detectors FRC teams actually weigh: **NAPA** (a fast sparse-grid CPU detector,
<https://www.chiefdelphi.com/t/introducing-napa-a-faster-apriltag-detector/525801>)
and **wpilib-d2** (`robotpy-apriltag`, WPILib's defaults — what robot code runs).
All five detectors run on one machine, so GPU-vs-CPU timing is honest. Every
detector is scored by **NAPA's own scorer** (`cpu_comparison.score_detail`: ID
match + best-of-8 cyclic/reflection corner alignment), so recall and corner
error are directly comparable.

Machine: **Quadro P620** GPU, **AMD EPYC 7302P** CPU. CPU backends: single
thread, pinned to one core, 1 warm-up + 10 timed interleaved reps/frame (NAPA's
protocol). GPU: `build/avk_batch` (device init once, one pipeline per resolution,
GPU-timestamp compute + wall-clock frame), decode single-threaded and pinned off
the CPU core. GPU corners come from AprilTag's decode, so a `-0.5 px` shift maps
them to the benchmark's pixel-centre convention (validated: mean offset 0.00 px).

Detectors: **vulkan-gpu** (this repo, `napa-benchmark` branch), **napa**
(`napa-vision` 0.5.1, `min_tag_px=32`), **wpilib-d2** (`robotpy-apriltag`
2026.2.2), **apriltag3-d1 / -d2** (upstream AprilTag 3 at full / half resolution).

### 10.1 Results

Latency is per-frame (CPU: median of 10 reps; GPU: upload+compute+wait+decode),
then percentiles across frames. Corner error in px on matched tags.

**main** (3,000 frames, 5,521 tags, 20 scenarios × 3 resolutions)

| detector | recall | false-ID | corner med / p95 / max px | p50 / p95 / p99 ms |
|---|---:|---:|---|---:|
| **vulkan-gpu** | 99.9% | 4 | 0.042 / 0.246 / 2.34 | 10.7 / 40.9 / 46.9 |
| napa | 99.9% | 1 | 0.020 / 0.092 / 3.39 | 0.42 / 2.49 / 3.55 |
| wpilib-d2 (FRC) | 86.2% | 0 | 0.099 / 0.303 / 7.55 | 19.4 / 71.1 / 81.5 |
| apriltag3-d1 | 99.9% | 3 | 0.042 / 0.246 / 2.35 | 103.3 / 369.7 / 408.0 |
| apriltag3-d2 | 99.7% | 3 | 0.043 / 0.305 / 2.03 | 25.9 / 90.5 / 105.3 |

**stress** (600 frames, 4,577 tags — tiny / dense / heavy-noise / noisy-clutter)

| detector | recall | false-ID | corner med / p95 / max px | p50 / p95 / p99 ms |
|---|---:|---:|---|---:|
| **vulkan-gpu** | 100.0% | 2 | 0.042 / 0.136 / 0.59 | 10.7 / 42.9 / 47.5 |
| napa | 95.9% | 0 | 0.020 / 0.105 / 0.32 | 0.93 / 3.78 / 4.52 |
| wpilib-d2 (FRC) | 65.8% | 0 | 0.113 / 0.226 / 0.47 | 19.1 / 69.5 / 74.8 |
| apriltag3-d1 | 100.0% | 1 | 0.042 / 0.136 / 0.59 | 99.2 / 376.6 / 408.6 |
| apriltag3-d2 | 97.1% | 1 | 0.041 / 0.128 / 2.53 | 27.7 / 93.5 / 111.8 |

**range** (900 frames, 900 tags — one tag per frame, black square 12→96 px)

| detector | recall | false-ID | corner med / p95 / max px | p50 / p95 / p99 ms |
|---|---:|---:|---|---:|
| **vulkan-gpu** | 99.9% | 0 | 0.051 / 0.269 / 0.69 | 4.6 / 5.5 / 5.9 |
| napa | 83.1% | 0 | 0.029 / 0.095 / 0.24 | 0.14 / 0.17 / 0.18 |
| wpilib-d2 (FRC) | 29.2% | 0 | 0.103 / 0.210 / 0.28 | 8.3 / 10.0 / 10.9 |
| apriltag3-d1 | 99.9% | 0 | 0.050 / 0.269 / 0.69 | 43.0 / 52.2 / 54.2 |
| apriltag3-d2 | 79.9% | 0 | 0.042 / 0.203 / 2.15 | 10.9 / 13.4 / 14.6 |

**Detection distance — recall (%) vs. tag size (black-square px), range sweep:**

| detector | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 32 | 56 | 64 | 72+ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **vulkan-gpu** | 98 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| apriltag3-d1 | 98 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| napa | 0 | 0 | 28 | 74 | 94 | 100 | 100 | 100 | 100 | 100 | 100 |
| apriltag3-d2 | 8 | 16 | 18 | 40 | 66 | 90 | 100 | 100 | 100 | 100 | 100 |
| wpilib-d2 (FRC) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 57 | 77 | 100 |

**GPU per-frame latency breakdown** (p50 / p95 ms):

| group | compute | upload+gpu+wait | cpu decode | total |
|---|---:|---:|---:|---:|
| main | 7.34 / 33.26 | 8.07 / 34.20 | 2.37 / 8.07 | 10.67 / 40.94 |
| stress | 6.80 / 33.62 | 7.53 / 34.56 | 2.98 / 9.63 | 10.74 / 42.94 |
| range | 3.11 / 3.78 | 3.73 / 4.40 | 0.80 / 1.21 | 4.57 / 5.46 |

**Per-frame latency ratio** (GPU time ÷ other; >1 ⇒ GPU faster; median over frames):

| | vs napa | vs wpilib-d2 | vs apriltag3-d1 | vs apriltag3-d2 |
|---|---:|---:|---:|---:|
| main | 0.06× | 1.69× | 8.68× | 2.27× |
| stress | 0.10× | 1.60× | 8.66× | 2.36× |
| range | 0.03× | 1.85× | 9.49× | 2.41× |

### 10.2 Cross-check vs. NAPA's published numbers

NAPA's published run (`results/2026-10-30_cpu_full_corpus`, **i7-12700H**) scored
the identical images. Our local recall matches theirs (main: napa 99.9/99.9,
wpilib-d2 86.2/86.2, apriltag3-d1 99.9/99.9), which confirms the scoring. Their
CPU latency is lower because the i7-12700H runs ~1.6–1.8× faster per core than
the EPYC 7302P (napa main p50 0.255 ms vs our 0.415 ms; apriltag3-d1 57.0 vs
103.3 ms). This is why all five detectors ran on one machine here.

### 10.3 Takeaways

- **NAPA is the fastest detector** — 10–33× faster per frame than the GPU, and
  faster than every CPU AprilTag mode, at matching corner accuracy (0.020 px).
  On a coprocessor with no usable GPU it beats the stock WPILib detector on both
  speed and recall.
- **NAPA's cost is range.** Its `min_tag_px=32` grid finds no tag below ~20 px:
  range recall 83.1%, a hard cliff (0% at 12–14 px), stress recall 95.9%.
- **The GPU holds the full-resolution recall ceiling** (99.9–100% everywhere,
  = apriltag3-d1, down to 12 px — ~2× NAPA's distance, ~4× wpilib-d2's) at
  ~9× the speed of full-res CPU AprilTag, same corner accuracy (same decoder).
  It is also ~1.7× faster than the WPILib detector and far higher recall.
- **wpilib-d2 (today's FRC default) is slow and short-range** — it finds no tag
  below a 56 px square, because WPILib decimates by 2 and requires a 300 px
  minimum cluster.
- **Net:** NAPA wins when CPU time is scarce and tags are large/near. The GPU
  wins when small-tag / long-range recall matters and a GPU is present, giving
  full-resolution recall at a latency the full-res CPU detector cannot reach,
  and spending no CPU.

### 10.4 Reproducing

The GPU side uses `build/avk_batch` (this repo). The napa / wpilib / apriltag3
backends and the scorer come from NAPA's own repo (not vendored — they need
`napa-vision`, `robotpy-apriltag`, and NAPA's `benchmarks/` modules):

```bash
# detectors + NAPA's harness
python3 -m venv ~/napa-bench/venv
~/napa-bench/venv/bin/pip install huggingface_hub numpy \
    napa-vision opencv-python-headless "robotpy-apriltag==2026.2.2"
git clone --depth 1 https://github.com/zwatx/NAPA.git ~/napa-bench/repo
# build the upstream AprilTag3 shim NAPA's harness loads, reusing this repo's apriltag:
SRC=build/_deps/apriltag-src; LIB=build/_deps/apriltag-build
cc -O3 -DNDEBUG -std=gnu99 -fPIC -shared -I"$SRC" \
   -o ~/napa-bench/repo/work/build/shims/libapriltag_shim.so \
   ~/napa-bench/repo/benchmarks/cpu_comparison_shims/apriltag_shim.c \
   -L"$LIB" -lapriltag -Wl,-rpath,"$LIB"

# CPU detectors over the corpus (single-thread, pinned; ~2 h: apriltag3-d1 @1080p is slow)
for g in main stress range; do
  taskset -c 2 ~/napa-bench/venv/bin/python ~/napa-bench/harness/run_cpu.py \
      --group "$g" --out ~/napa-bench/out/cpu_${g}.csv
done

# GPU over the same frames (manifest = one PNG path per line, metadata order)
for g in main stress range; do
  taskset -c 4-15 build/avk_batch ~/napa-bench/out/manifest_${g}.txt --device 0 \
      --iterations 10 --warmup 2 --decode-threads 1 > ~/napa-bench/out/gpu_${g}.jsonl
done

~/napa-bench/venv/bin/python ~/napa-bench/harness/score.py   # prints the tables above
```

`run_cpu.py` and `score.py` live in `~/napa-bench/harness/` on the reference box.
`avk_batch` keeps one pipeline per resolution at a time (rebuild on change), so
peak VRAM is a single resolution — a 2 GB card cannot hold all three at once.
