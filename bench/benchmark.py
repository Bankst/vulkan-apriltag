#!/usr/bin/env python3
"""Run GPU and/or CPU AprilTag detectors against the NAPA benchmark dataset
and compare detections to ground truth.

Usage:
    python3 bench/benchmark.py [options]

Options:
    --split SPLIT       Dataset split: main, stress, range (default: main)
    --scenario PATTERN  Filter scenarios by substring (e.g. "blur", "many-tags")
    --gpu-bin PATH      Path to vulktan_apriltag binary (default: build/vulktan_apriltag)
    --cpu-bin PATH      Path to cpu_bench binary (default: build/cpu_bench)
    --device DEV        GPU device index or name (default: 0)
    --iterations N      Timing iterations per image (default: 5)
    --warmup N          Warmup iterations (default: 2)
    --cpu-only          Only run CPU detector
    --gpu-only          Only run GPU detector
    --decimate D        CPU decimation factor (default: 1.0)
    --threads T         CPU threads (default: 4)
    --limit N           Max images to process (default: all)
    --dataset-dir DIR   Dataset root (default: bench/dataset)
    --summary           Print summary only, skip per-image output
    --json              Output results as JSON
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_gpu_output(text):
    """Parse vulktan_apriltag stdout into detections and timing."""
    detections = []
    timing = {}
    for line in text.splitlines():
        m = re.match(
            r"\s+id (\d+) \(hamming (\d+), margin ([\d.]+)\) "
            r"centre \(([\d.-]+), ([\d.-]+)\) corners"
            r" \(([\d.-]+), ([\d.-]+)\)"
            r" \(([\d.-]+), ([\d.-]+)\)"
            r" \(([\d.-]+), ([\d.-]+)\)"
            r" \(([\d.-]+), ([\d.-]+)\)",
            line,
        )
        if m:
            g = m.groups()
            detections.append({
                "id": int(g[0]),
                "hamming": int(g[1]),
                "margin": float(g[2]),
                "center": [float(g[3]), float(g[4])],
                "corners": [
                    [float(g[5]), float(g[6])],
                    [float(g[7]), float(g[8])],
                    [float(g[9]), float(g[10])],
                    [float(g[11]), float(g[12])],
                ],
            })
            continue
        m = re.match(r"\s+([\w :+()]+?)\s+([\d.]+) / ([\d.]+)", line)
        if m:
            timing[m.group(1).strip()] = {
                "min": float(m.group(2)),
                "median": float(m.group(3)),
            }
    return detections, timing


def parse_cpu_output(text):
    """Parse cpu_bench stdout into detections and timing."""
    detections = []
    timing = {}
    for line in text.splitlines():
        m = re.match(
            r"\s+id (\d+) \(hamming (\d+), margin ([\d.]+)\) "
            r"centre \(([\d.-]+), ([\d.-]+)\) "
            r"corners \(([\d.-]+), ([\d.-]+)\)"
            r" \(([\d.-]+), ([\d.-]+)\)"
            r" \(([\d.-]+), ([\d.-]+)\)"
            r" \(([\d.-]+), ([\d.-]+)\)",
            line,
        )
        if m:
            g = m.groups()
            detections.append({
                "id": int(g[0]),
                "hamming": int(g[1]),
                "margin": float(g[2]),
                "center": [float(g[3]), float(g[4])],
                "corners": [
                    [float(g[5]), float(g[6])],
                    [float(g[7]), float(g[8])],
                    [float(g[9]), float(g[10])],
                    [float(g[11]), float(g[12])],
                ],
            })
            continue
        m = re.match(r"\s+detect\+decode median/min: ([\d.]+) / ([\d.]+)", line)
        if m:
            timing["detect+decode"] = {
                "median": float(m.group(1)),
                "min": float(m.group(2)),
            }
    return detections, timing


def corner_distance(det_corners, gt_corners, debias=0.0):
    """Mean Euclidean distance between detected and ground-truth corners,
    trying all 4 rotations in both winding orders and picking the best
    match (corner ordering conventions differ between detectors)."""
    best = float("inf")
    orderings = [gt_corners, gt_corners[::-1]]
    for gt in orderings:
        for rot in range(4):
            total = 0.0
            for k in range(4):
                gx, gy = gt[(k + rot) % 4]
                dx, dy = det_corners[k]
                dx -= debias
                dy -= debias
                total += math.sqrt((dx - gx) ** 2 + (dy - gy) ** 2)
            best = min(best, total / 4.0)
    return best


def evaluate(detections, ground_truth, debias=0.0):
    """Compare detections against ground truth.

    Returns dict with tp, fp, fn, and per-tag corner errors.
    """
    gt_ids = set(ground_truth["tag_ids"])
    det_ids = {d["id"] for d in detections}

    tp_ids = gt_ids & det_ids
    fp_ids = det_ids - gt_ids
    fn_ids = gt_ids - det_ids

    gt_corners_by_id = {}
    for i, tid in enumerate(ground_truth["tag_ids"]):
        gt_corners_by_id[tid] = ground_truth["corners"][i]

    corner_errors = []
    for d in detections:
        if d["id"] in gt_corners_by_id:
            err = corner_distance(d["corners"], gt_corners_by_id[d["id"]], debias)
            corner_errors.append({"id": d["id"], "error_px": err})

    return {
        "tp": len(tp_ids),
        "fp": len(fp_ids),
        "fn": len(fn_ids),
        "fp_ids": sorted(fp_ids),
        "fn_ids": sorted(fn_ids),
        "corner_errors": corner_errors,
    }


def run_detector(binary, image_path, args):
    """Run a detector binary and return stdout."""
    cmd = [binary, str(image_path)] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except FileNotFoundError:
        return "", f"binary not found: {binary}", -1


def load_metadata(dataset_dir, split):
    """Load metadata.jsonl for a split."""
    meta_path = Path(dataset_dir) / "data" / split / "metadata.jsonl"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run bench/download_dataset.sh first.",
              file=sys.stderr)
        sys.exit(1)
    entries = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description="NAPA AprilTag benchmark runner")
    parser.add_argument("--split", default="main")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--gpu-bin", default="build/vulktan_apriltag")
    parser.add_argument("--cpu-bin", default="build/cpu_bench")
    parser.add_argument("--device", default="0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--decimate", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-dir", default="bench/dataset")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debias", type=float, default=0.0,
                        help="Shift detected corners by -N px (corner-convention offset)")
    args = parser.parse_args()

    entries = load_metadata(args.dataset_dir, args.split)
    if args.scenario:
        entries = [e for e in entries if args.scenario in e["scenario"]]
    if args.limit:
        entries = entries[: args.limit]

    if not entries:
        print("No matching images found.", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.dataset_dir) / "data" / args.split
    run_gpu = not args.cpu_only
    run_cpu = not args.gpu_only

    results = {"split": args.split, "images": len(entries), "gpu": None, "cpu": None}

    for label, do_run, binary, make_args, parse_fn in [
        (
            "GPU",
            run_gpu,
            args.gpu_bin,
            lambda: [
                "--device", args.device,
                "--iterations", str(args.iterations),
                "--warmup", str(args.warmup),
            ],
            parse_gpu_output,
        ),
        (
            "CPU",
            run_cpu,
            args.cpu_bin,
            lambda: [
                str(args.decimate),
                str(args.iterations),
                str(args.warmup),
                str(args.threads),
            ],
            parse_cpu_output,
        ),
    ]:
        if not do_run:
            continue

        if not Path(binary).exists():
            print(f"Warning: {binary} not found, skipping {label}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f" {label} detector — split={args.split}, {len(entries)} images")
        print(f"{'='*60}")

        total_tp = total_fp = total_fn = 0
        all_corner_errors = []
        per_scenario = {}
        errors = 0

        for i, entry in enumerate(entries):
            img_path = data_dir / entry["file_name"]
            if not img_path.exists():
                print(f"  Missing: {entry['file_name']}", file=sys.stderr)
                errors += 1
                continue

            stdout, stderr, rc = run_detector(binary, img_path, make_args())

            if rc != 0:
                if not args.summary:
                    print(f"  FAIL: {entry['file_name']}: {stderr.strip()[:80]}")
                errors += 1
                continue

            detections, timing = parse_fn(stdout)
            ev = evaluate(detections, entry, debias=args.debias)

            total_tp += ev["tp"]
            total_fp += ev["fp"]
            total_fn += ev["fn"]
            all_corner_errors.extend(ev["corner_errors"])

            scenario = entry["scenario"]
            if scenario not in per_scenario:
                per_scenario[scenario] = {"tp": 0, "fp": 0, "fn": 0, "count": 0, "errors": []}
            per_scenario[scenario]["tp"] += ev["tp"]
            per_scenario[scenario]["fp"] += ev["fp"]
            per_scenario[scenario]["fn"] += ev["fn"]
            per_scenario[scenario]["count"] += 1
            per_scenario[scenario]["errors"].extend(ev["corner_errors"])

            if not args.summary and (ev["fp"] > 0 or ev["fn"] > 0):
                print(
                    f"  {entry['file_name']}: "
                    f"tp={ev['tp']} fp={ev['fp']} fn={ev['fn']}"
                    + (f" fp_ids={ev['fp_ids']}" if ev["fp_ids"] else "")
                    + (f" fn_ids={ev['fn_ids']}" if ev["fn_ids"] else "")
                )

            if not args.summary and (i + 1) % 100 == 0:
                print(f"  ... processed {i+1}/{len(entries)}")

        total_gt = total_tp + total_fn
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        recall = total_tp / total_gt if total_gt else 0
        errs_sorted = sorted(e["error_px"] for e in all_corner_errors) if all_corner_errors else []
        mean_corner_err = sum(errs_sorted) / len(errs_sorted) if errs_sorted else float("nan")
        median_corner_err = errs_sorted[len(errs_sorted) // 2] if errs_sorted else float("nan")
        p95_corner_err = errs_sorted[int(len(errs_sorted) * 0.95)] if errs_sorted else float("nan")
        max_corner_err = errs_sorted[-1] if errs_sorted else float("nan")

        print(f"\n--- {label} Summary ---")
        print(f"  Images: {len(entries)}  Errors: {errors}")
        print(f"  Tags: TP={total_tp} FP={total_fp} FN={total_fn}")
        print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
        if errs_sorted:
            print(f"  Corner error (px): median={median_corner_err:.4f}  "
                  f"mean={mean_corner_err:.4f}  p95={p95_corner_err:.4f}  "
                  f"max={max_corner_err:.4f}")
        if args.debias != 0.0:
            print(f"  (debias: {args.debias} px applied to detected corners)")

        print(f"\n  Per-scenario breakdown:")
        print(f"  {'Scenario':<25s} {'Images':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Recall':>7s} {'MeanErr':>8s}")
        for scen in sorted(per_scenario):
            s = per_scenario[scen]
            gt = s["tp"] + s["fn"]
            rec = s["tp"] / gt if gt else 0
            merr = (
                sum(e["error_px"] for e in s["errors"]) / len(s["errors"])
                if s["errors"]
                else float("nan")
            )
            print(
                f"  {scen:<25s} {s['count']:>6d} {s['tp']:>5d} {s['fp']:>5d} "
                f"{s['fn']:>5d} {rec:>7.4f} {merr:>8.2f}"
            )

        results[label.lower()] = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "mean_corner_error_px": mean_corner_err,
            "per_scenario": {
                k: {"tp": v["tp"], "fp": v["fp"], "fn": v["fn"], "count": v["count"]}
                for k, v in per_scenario.items()
            },
        }

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
