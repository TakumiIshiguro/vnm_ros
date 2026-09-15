#!/usr/bin/env python3
"""Correct a constant yaw lag in a collected dataset.

The sim bags' localization publishes a pose whose yaw trails its own position
stream by ~0.7 s during turns (real bags: ~0.14 s). Because the teacher
waypoints are expressed in the robot frame via ``yaw[i]``, that lag rotates
every future waypoint towards the turn by ``omega * tau``, which inflates the
labelled turn angle by a constant factor ``1 + 2*tau/T`` (T = the prediction
horizon in seconds). This tool estimates tau per trajectory and rewrites
``traj_data.pkl`` with the yaw resampled at ``t + tau``; images are symlinked
rather than copied.
"""
import argparse
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np

from vnm_ros.datasets.trajectory_dataset import discover_trajectory_names
from vnm_ros.datasets.dataset_utils import numeric_image_files
from vnm_ros.utils.config import package_root, resolve_path


MIN_STEP_M = 0.02
OFFSETS = np.arange(-1, 7)


def wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def turning_samples(position, yaw, cmd_dir, horizon, min_turn_deg):
    samples = []
    limit = len(position) - horizon - int(OFFSETS.max()) - 1
    for current in range(1, limit):
        if int(np.argmax(cmd_dir[current, :3])) == 0:
            continue
        net = wrap(yaw[current + horizon] - yaw[current])
        if abs(math.degrees(net)) < min_turn_deg:
            continue
        # Centred difference so the direction of motion is dated at `current`
        # rather than half a sample later.
        step = position[current + 1] - position[current - 1]
        if np.linalg.norm(step) < 2.0 * MIN_STEP_M:
            continue
        samples.append((current, math.copysign(1.0, net)))
    return samples


def yaw_lag_seconds(position, yaw, samples, sample_dt):
    """Offset at which the recorded yaw matches the actual direction of motion."""
    bias = []
    for offset in OFFSETS:
        errors = []
        for current, sign in samples:
            step = position[current + 1] - position[current - 1]
            motion = math.atan2(step[1], step[0])
            errors.append(math.degrees(wrap(motion - yaw[current + offset])) * sign)
        bias.append(float(np.mean(errors)))
    bias = np.asarray(bias)
    crossings = np.where(np.diff(np.sign(bias)))[0]
    if len(crossings) == 0:
        return None, bias
    index = int(crossings[0])
    fraction = bias[index] / (bias[index] - bias[index + 1])
    return float((OFFSETS[index] + fraction) * sample_dt), bias


def label_turn_ratio(position, yaw, cmd_dir, horizon, min_turn_deg):
    """Mean of (labelled endpoint bearing) / (geometric bearing = net yaw / 2)."""
    ratios = []
    for current in range(len(position) - horizon - 1):
        if int(np.argmax(cmd_dir[current, :3])) == 0:
            continue
        net = wrap(yaw[current + horizon] - yaw[current])
        if abs(math.degrees(net)) < min_turn_deg:
            continue
        rotation = np.array(
            [
                [math.cos(yaw[current]), math.sin(yaw[current])],
                [-math.sin(yaw[current]), math.cos(yaw[current])],
            ]
        )
        local = (position[current + 1 : current + 1 + horizon] - position[current]) @ rotation.T
        bearing = math.atan2(local[-1, 1], local[-1, 0])
        ratios.append(bearing / (net / 2.0))
    return float(np.mean(ratios)) if ratios else float("nan")


def shift_yaw(yaw, shift_samples):
    """Resample yaw at index + shift_samples, interpolating the unwrapped angle."""
    unwrapped = np.unwrap(np.asarray(yaw, dtype=np.float64))
    source = np.arange(len(unwrapped), dtype=np.float64)
    target = source + shift_samples
    return wrap(np.interp(target, source, unwrapped)).astype(np.float32)


def link_images(source_dir, target_dir, count):
    files = numeric_image_files(source_dir)
    if len(files) < count:
        raise ValueError(f"{source_dir}: images={len(files)} < samples={count}")
    for name in files[:count]:
        link = os.path.join(target_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(os.path.join(source_dir, name)), link)


def load_trajectory(source_dir, args):
    with open(os.path.join(source_dir, "traj_data.pkl"), "rb") as f:
        data = pickle.load(f)
    position = np.asarray(data["position"], dtype=np.float32)
    yaw = np.asarray(data["yaw"], dtype=np.float32).reshape(-1)
    cmd_dir = np.asarray(data["cmd_dir"], dtype=np.float32)
    metadata = dict(data.get("metadata", {}))
    sample_dt = float(args.sample_dt or metadata.get("sample_dt", 0.25))
    return position, yaw, cmd_dir, metadata, sample_dt


def estimate_trajectory_lag(source_dir, args):
    """Per-trajectory yaw lag, or None when it holds too few turning samples."""
    position, yaw, cmd_dir, _, sample_dt = load_trajectory(source_dir, args)
    samples = turning_samples(position, yaw, cmd_dir, args.horizon, args.min_turn_deg)
    if len(samples) < args.min_turn_samples:
        return None, len(samples)
    estimated, _ = yaw_lag_seconds(position, yaw, samples, sample_dt)
    return estimated, len(samples)


def fix_trajectory(source_dir, target_dir, tau, estimated, turning_count, args):
    position, yaw, cmd_dir, metadata, sample_dt = load_trajectory(source_dir, args)

    shift_samples = tau / sample_dt
    trimmed = len(position) - int(math.ceil(shift_samples))
    if trimmed <= args.horizon + args.context_size:
        raise ValueError(f"{source_dir}: trajectory too short after trimming")

    fixed_yaw = shift_yaw(yaw, shift_samples)[:trimmed]
    fixed_position = position[:trimmed]
    fixed_cmd_dir = cmd_dir[:trimmed]

    before = label_turn_ratio(position, yaw, cmd_dir, args.horizon, args.min_turn_deg)
    after = label_turn_ratio(
        fixed_position, fixed_yaw, fixed_cmd_dir, args.horizon, args.min_turn_deg
    )

    metadata["yaw_lag_correction_seconds"] = float(tau)
    metadata["yaw_lag_estimated_seconds"] = (
        float(estimated) if estimated is not None else None
    )
    metadata["yaw_lag_source_dataset"] = os.path.abspath(source_dir)
    metadata["yaw_lag_turning_samples"] = int(turning_count)

    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(
            {
                "position": fixed_position,
                "yaw": fixed_yaw,
                "cmd_dir": fixed_cmd_dir,
                "metadata": metadata,
            },
            f,
        )
    link_images(source_dir, target_dir, trimmed)
    print(
        f"{os.path.basename(target_dir)}: tau={tau:.3f}s "
        f"({shift_samples:.2f} samples) turning_samples={turning_count} "
        f"samples {len(position)}->{trimmed} "
        f"label_turn_ratio {before:.3f}->{after:.3f}"
    )
    return tau, before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="dataset dir, e.g. dataset/bags_all_train")
    parser.add_argument("--output", required=True, help="dataset dir to create")
    parser.add_argument("--model-type", default="nomad")
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="yaw lag in seconds; estimated per trajectory when omitted",
    )
    parser.add_argument("--sample-dt", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=8, help="len_traj_pred")
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--min-turn-deg", type=float, default=15.0)
    parser.add_argument("--min-turn-samples", type=int, default=20)
    args = parser.parse_args()

    root = package_root()
    source_root = os.path.join(resolve_path(args.source, root), args.model_type)
    target_root = os.path.join(resolve_path(args.output, root), args.model_type)
    names = discover_trajectory_names(source_root)
    if not names:
        raise ValueError(f"No trajectories found in {source_root}")
    print(f"correcting {len(names)} trajectories: {source_root} -> {target_root}")

    # Estimate first, so a trajectory without enough turns (a straight-only
    # corridor run) can fall back to the mean lag of the ones that have them.
    estimates = {}
    counts = {}
    for name in names:
        estimates[name], counts[name] = estimate_trajectory_lag(
            os.path.join(source_root, name), args
        )
    measured = [v for v in estimates.values() if v is not None]
    if args.tau is None and not measured:
        raise ValueError(
            f"{source_root}: no trajectory has >= {args.min_turn_samples} turning "
            "samples; pass --tau explicitly"
        )
    fallback = args.tau if args.tau is not None else float(np.mean(measured))

    taus = []
    for name in names:
        tau = args.tau if args.tau is not None else estimates[name]
        if tau is None:
            tau = fallback
            print(
                f"{name}: only {counts[name]} turning samples, "
                f"using mean tau={fallback:.3f}s"
            )
        tau, _, after = fix_trajectory(
            os.path.join(source_root, name),
            os.path.join(target_root, name),
            tau,
            estimates[name],
            counts[name],
            args,
        )
        taus.append(tau)
    print(
        f"done: mean tau={np.mean(taus):.3f}s "
        f"(min={np.min(taus):.3f} max={np.max(taus):.3f})"
    )


if __name__ == "__main__":
    main()
