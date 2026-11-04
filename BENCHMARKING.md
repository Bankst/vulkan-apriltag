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
