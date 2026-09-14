#!/usr/bin/env python3
"""Interactive two-camera FoundationPose tracking and diagnostics.

This is an evaluation adapter, not the production ROS driver.  It opens both
configured D435i devices, initializes one FoundationPose estimator per camera,
converts both poses into the calibrated rig frame, and feeds both observations
to one shared Kalman filter.  Raw camera, rig, fused, and timing records are
saved as both JSONL and NPZ files for offline debugging.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    values = tuple(int(item) for item in value.split(","))
    if len(values) != 4 or values[2] < 2 or values[3] < 2:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height with width/height >= 2")
    return values


def make_mask(shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x, y, roi_width, roi_height = roi
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + roi_width), min(height, y + roi_height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def pose_to_6d(pose) -> np.ndarray:
    """Convert either a torch or NumPy object transform to the 6D KF state."""
    pose_np = pose.detach().cpu().numpy() if hasattr(pose, "detach") else np.asarray(pose)
    if pose_np.ndim == 3:
        pose_np = pose_np[0]
    return np.r_[
        pose_np[:3, 3],
        Rotation.from_matrix(pose_np[:3, :3]).as_euler("xyz"),
    ]


def sixd_to_pose(values) -> np.ndarray:
    """Convert the 6D local-filter state to a float32 FoundationPose pose."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("pose state must be a finite six-element vector")
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix().astype(
        np.float32
    )
    pose[:3, 3] = values[:3].astype(np.float32)
    return pose


def adjust_pose_to_image_point(pose, K, x, y):
    """Move a centered-mesh pose origin to a 2D tracker center point."""
    batched = pose.ndim == 3
    output = pose[0].clone() if batched else pose.clone()
    z = output[2, 3]
    output[0, 3] = (float(x) - float(K[0, 2])) * z / float(K[0, 0])
    output[1, 3] = (float(y) - float(K[1, 2])) * z / float(K[1, 1])
    return output.unsqueeze(0) if batched else output


def quaternion_matrix_wxyz(values: list[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_wxyz must contain four finite values")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("quaternion_wxyz has zero norm")
    w, x, y, z = quaternion / norm
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return transform


def load_rig_extrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        camera_0 = document["T_rig_camera_0"]
        camera_1 = document["T_rig_camera_1"]
        t0 = np.eye(4, dtype=np.float64)
        t0[:3, :3] = quaternion_matrix_wxyz(
            camera_0["quaternion_wxyz"]
        )[:3, :3]
        t0[:3, 3] = np.asarray(camera_0["translation_xyz_m"], dtype=np.float64)
        t1 = quaternion_matrix_wxyz(camera_1["quaternion_wxyz"])
        t1[:3, 3] = np.asarray(camera_1["translation_xyz_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid rig calibration {path}: {exc}") from exc
    return t0, t1


def create_estimator(mesh_path: Path, mesh_scale: float, debug_dir: Path, cube_symmetry=True):
    import trimesh

    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import dr, draw_posed_3d_box, draw_xyz_axis, trimesh_add_pure_colored_texture

    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.apply_scale(mesh_scale)
    try:
        mesh = trimesh_add_pure_colored_texture(
            mesh, color=np.asarray([0, 159, 237], dtype=np.uint8), resolution=10
        )
    except Exception as exc:  # pragma: no cover - depends on optional texture path
        logging.warning("mesh texture fallback failed: %s", exc)
    mesh._data["vertices"] = np.asarray(mesh.vertices, dtype=np.float32)
    mesh._cache.clear()
    # A plain cube has 24 geometrically equivalent proper rotations.  Passing
    # these to FoundationPose keeps its registration grid aware of the object
    # symmetry; the live loop additionally canonicalizes the returned branch
    # against the rig prior before fusing two cameras.
    extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    is_cube_mesh = bool(
        np.allclose(extents, extents[0], rtol=0.0, atol=1e-6)
    )
    symmetry_tfs = cube_symmetry_transforms() if cube_symmetry and is_cube_mesh else None
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        symmetry_tfs=symmetry_tfs,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir=str(debug_dir),
    )
    if estimator.mesh is not None:
        estimator.mesh._data["vertices"] = np.asarray(
            estimator.mesh.vertices, dtype=np.float32
        )
        estimator.mesh._cache.clear()
    estimator.diameter = float(estimator.diameter)
    # The FP++-compatible path keeps FoundationPose's original one-branch
    # rotation grid, but its output still needs cube branch canonicalization
    # before the two camera measurements can be compared in the rig frame.
    estimator.is_cube_mesh = is_cube_mesh
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
    return estimator, np.asarray(to_origin), bbox, draw_posed_3d_box, draw_xyz_axis


def cube_symmetry_transforms() -> np.ndarray:
    """Return the 24 proper rotational symmetries of a centered cube."""
    transforms = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float64)
            rotation[np.arange(3), permutation] = signs
            if np.linalg.det(rotation) <= 0.0:
                continue
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transforms.append(transform)
    return np.asarray(transforms, dtype=np.float64)


def start_camera(rs, serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    # D435i devices can expose independent hardware clocks.  Global Time makes
    # the color timestamps comparable across the two USB devices.  If a
    # firmware does not expose this option, wait_pair() falls back to host
    # monotonic timestamps and still acquires both frames concurrently.
    global_time_enabled = False
    try:
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)
                global_time_enabled = (
                    sensor.get_option(rs.option.global_time_enabled) > 0.5
                ) or global_time_enabled
    except Exception as exc:
        logging.warning("could not enable RealSense Global Time for %s: %s", serial, exc)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    K = np.asarray(
        [[intrinsics.fx, 0.0, intrinsics.ppx],
         [0.0, intrinsics.fy, intrinsics.ppy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return pipeline, rs.align(rs.stream.color), depth_scale, K, global_time_enabled


def wait_one(pipeline, align, use_device_time):
    """Acquire one aligned frameset and timestamp its host acquisition."""
    # If inference takes longer than the camera period, the SDK queue can
    # contain a backlog of old frames.  Drain it and process only the newest
    # frameset so display latency follows the camera rather than the queue.
    raw_frames = pipeline.wait_for_frames()
    while True:
        newer = pipeline.poll_for_frames()
        if not newer:
            break
        raw_frames = newer
    frames = align.process(raw_frames)
    # Prefer the RealSense device timestamp.  It is the timestamp of the
    # image exposure and is comparable between cameras when Global Time is
    # enabled.  Host monotonic time remains the fallback for firmware that
    # reports no usable device timestamp.
    host_stamp_ns = time.monotonic_ns()
    color = frames.get_color_frame()
    device_stamp_ms = float(color.get_timestamp()) if color else float("nan")
    frame_number = int(color.get_frame_number()) if color else -1
    if use_device_time and np.isfinite(device_stamp_ms) and device_stamp_ms > 0.0:
        return (
            frames,
            int(round(device_stamp_ms * 1e6)),
            host_stamp_ns,
            frame_number,
            int(round(device_stamp_ms * 1e6)),
        )
    return frames, host_stamp_ns, host_stamp_ns, frame_number, -1


def wait_pair(
    executor,
    pipeline0,
    align0,
    use_device_time0,
    pipeline1,
    align1,
    use_device_time1,
    max_skew_ns,
):
    """Wait for both devices concurrently.

    Calling the two blocking waits sequentially can make an otherwise valid
    stereo pair appear tens of milliseconds apart.  The host timestamp is
    taken immediately after each wait returns, while the two waits themselves
    run concurrently in separate threads.
    """
    future0 = executor.submit(wait_one, pipeline0, align0, use_device_time0)
    future1 = executor.submit(wait_one, pipeline1, align1, use_device_time1)
    item0, item1 = future0.result(), future1.result()
    # A device timestamp is only comparable if both cameras have Global Time.
    # Mixed firmware falls back to the concurrent host-return timestamps.
    if use_device_time0 and use_device_time1 and abs(item0[1] - item1[1]) <= max_skew_ns:
        return item0, item1
    # Mixed firmware or an unusable device clock: use the host return time,
    # while retaining the frame number and raw device timestamp for logging.
    return (
        item0[0], item0[2], item0[2], item0[3], item0[4]
    ), (
        item1[0], item1[2], item1[2], item1[3], item1[4]
    )


def draw_pose_overlay(
    bgr: np.ndarray,
    camera_pose: np.ndarray,
    to_origin: np.ndarray,
    bbox: np.ndarray,
    K: np.ndarray,
    draw_posed_3d_box,
    draw_xyz_axis,
    label: str,
    line_color: tuple[int, int, int],
    draw_axis: bool = False,
) -> np.ndarray:
    """Render one pose in a camera image.

    ``camera_pose`` is the public FoundationPose transform (the object frame
    used by ``track_one``).  Keeping this helper generic lets the live view
    show the two raw camera estimates as well as the filtered rig estimate.
    """
    center_pose = camera_pose @ np.linalg.inv(to_origin)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = draw_posed_3d_box(
        K, img=rgb, ob_in_cam=center_pose, bbox=bbox,
        line_color=line_color, linewidth=2,
    )
    if draw_axis:
        rgb = draw_xyz_axis(
            rgb,
            ob_in_cam=center_pose,
            scale=0.10,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    xyz = camera_pose[:3, 3]
    # Text is drawn after conversion and therefore uses OpenCV BGR colors.
    text_color = (line_color[2], line_color[1], line_color[0])
    cv2.putText(
        output,
        f"{label}  t=[{xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}]m",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color,
        2,
        cv2.LINE_AA,
    )
    return output


def _finite_matrix(value) -> np.ndarray | None:
    """Return a finite 4x4 matrix or ``None`` for a failed measurement."""
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _json_matrix(value):
    matrix = _finite_matrix(value)
    return None if matrix is None else matrix.tolist()


def _json_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def pose_jump_metrics(previous, current) -> tuple[float, float]:
    """Return translation jump (m) and rotation jump (degrees)."""
    if previous is None or current is None:
        return 0.0, 0.0
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.shape != (4, 4) or current.shape != (4, 4):
        return float("inf"), float("inf")
    translation_m = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    try:
        from scipy.spatial.transform import Rotation

        relative = previous[:3, :3].T @ current[:3, :3]
        rotation_deg = float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))
    except (ValueError, np.linalg.LinAlgError):
        rotation_deg = float("inf")
    return translation_m, rotation_deg


def gate_pose_jump(
    candidate,
    reference,
    pending,
    pending_count: int,
    max_translation_m: float,
    max_rotation_deg: float,
):
    """Reject an isolated jump but relock to a stable new pose branch."""
    jump = pose_jump_metrics(reference, candidate)
    if candidate is None:
        return None, pending, pending_count, jump, False
    if jump[0] <= max_translation_m and jump[1] <= max_rotation_deg:
        return candidate, None, 0, jump, False

    pending_jump = pose_jump_metrics(pending, candidate)
    stable_pending = (
        pending is not None
        and pending_jump[0] <= max_translation_m
        and pending_jump[1] <= max_rotation_deg
    )
    next_count = pending_count + 1 if stable_pending else 1
    candidate_copy = np.asarray(candidate, dtype=np.float64).copy()
    if next_count >= 2:
        return candidate, None, 0, jump, True
    return None, candidate_copy, next_count, jump, False


def pose_disagreement(left, right) -> tuple[float, float]:
    """Return translation (m) and rotation (degrees) disagreement."""
    left = _finite_matrix(left)
    right = _finite_matrix(right)
    if left is None or right is None:
        return float("inf"), float("inf")
    translation_m = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    try:
        rotation_deg = float(np.rad2deg(
            Rotation.from_matrix(left[:3, :3].T @ right[:3, :3]).magnitude()
        ))
    except (ValueError, np.linalg.LinAlgError):
        rotation_deg = float("inf")
    return translation_m, rotation_deg


def public_cube_symmetry_transforms(model_center: np.ndarray) -> np.ndarray:
    """Convert centered-mesh cube symmetries to the public mesh frame."""
    center = np.asarray(model_center, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("model_center must be a finite three-element vector")
    to_center = np.eye(4, dtype=np.float64)
    to_center[:3, 3] = -center
    return np.asarray(
        [np.linalg.inv(to_center) @ symmetry @ to_center for symmetry in cube_symmetry_transforms()],
        dtype=np.float64,
    )


def canonicalize_cube_pose(
    pose: np.ndarray | None,
    reference: np.ndarray | None,
    model_center: np.ndarray,
) -> np.ndarray | None:
    """Choose the cube-equivalent branch nearest a trusted reference pose."""
    if pose is None or reference is None:
        return pose
    pose = _finite_matrix(pose)
    reference = _finite_matrix(reference)
    if pose is None or reference is None:
        return pose
    candidates = [pose @ symmetry for symmetry in public_cube_symmetry_transforms(model_center)]
    # All 24 candidates describe the same cube geometry.  Keep its coordinate
    # frame continuous by choosing the smallest SO(3) change first.  A weighted
    # translation-plus-rotation score can prefer a 90/180-degree branch merely
    # because the STL origin is at a corner instead of the cube centre.
    def score(candidate: np.ndarray) -> tuple[float, float]:
        translation = np.linalg.norm(candidate[:3, 3] - reference[:3, 3])
        relative = reference[:3, :3].T @ candidate[:3, :3]
        try:
            rotation = Rotation.from_matrix(relative).magnitude()
        except (ValueError, np.linalg.LinAlgError):
            rotation = np.pi
        return float(rotation), float(translation)
    return min(candidates, key=score)


def estimator_is_cube(estimator) -> bool:
    """Whether an estimator tracks a geometrically symmetric cube mesh."""
    if bool(getattr(estimator, "is_cube_mesh", False)):
        return True
    symmetries = getattr(estimator, "symmetry_tfs", None)
    try:
        return symmetries is not None and int(symmetries.shape[0]) == 24
    except (AttributeError, TypeError, ValueError):
        return False


def choose_roi(window: str, bgr: np.ndarray, requested):
    if requested is not None:
        return requested
    x, y, width, height = cv2.selectROI(window, bgr, showCrosshair=True)
    if width < 2 or height < 2:
        raise RuntimeError(f"no valid ROI selected for {window}")
    return int(x), int(y), int(width), int(height)


def register(estimator, K, rgb, depth, roi, refine_iter):
    mask = make_mask(rgb.shape[:2], roi)
    if np.count_nonzero((mask > 0) & (depth > 1e-4)) < 20:
        raise RuntimeError("selected ROI contains too few valid depth pixels")
    pose = estimator.register(
        K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=refine_iter
    )
    return np.asarray(pose, dtype=np.float64), mask


def public_to_internal(estimator, public_pose, torch):
    """Convert a public object pose back to FoundationPose.pose_last format."""
    centered = np.asarray(estimator.get_tf_to_centered_mesh().detach().cpu())
    internal = np.asarray(public_pose) @ np.linalg.inv(centered)
    return torch.as_tensor(internal, device="cuda", dtype=torch.float32).unsqueeze(0)


def track_one_timed(estimator, rgb, depth, K, iteration):
    """Run one estimator and return ``(pose, elapsed_ms, error)``.

    The wrapper keeps exception handling inside each worker, so a failed view
    cannot block the other view's result or leave an unobserved Future error.
    """
    started_ns = time.monotonic_ns()
    try:
        pose = estimator.track_one(rgb, depth, K, iteration)
        return pose, (time.monotonic_ns() - started_ns) / 1e6, None
    except Exception as exc:  # pragma: no cover - hardware/model dependent
        return None, (time.monotonic_ns() - started_ns) / 1e6, str(exc)


class LocalPlusPlus:
    """Single-view FP++ prior; fusion receives the raw FP measurement.

    Local filtered outputs are never fused as independent measurements. Each
    view keeps its own Cutie memory and centered-mesh KF state. Rig rejection
    must not overwrite these local priors.
    """

    def __init__(self, estimator, rgb, mask, stamp, use_2d, use_kf):
        self.tracker = None
        if use_2d:
            from VOT import Cutie
            self.tracker = Cutie()
            self.tracker.initialize(rgb, init_info={"mask": (mask > 0).astype(np.uint8)})
        from utils.kalman_filter_6d import LegacyKalmanFilter6D

        self.kf = LegacyKalmanFilter6D(0.05) if use_kf else None
        self.mean = self.covariance = None
        if self.kf is not None:
            self.mean, self.covariance = self.kf.initiate(
                pose_to_6d(estimator.pose_last)
            )
        self.stamp = stamp
        self.diagnostic = {}

    def track(self, estimator, rgb, depth, K, iteration):
        import torch

        started = time.monotonic_ns()
        previous_pose = estimator.pose_last.clone()
        previous_mean = None if self.mean is None else self.mean.copy()
        previous_cov = None if self.covariance is None else self.covariance.copy()
        self.diagnostic = {"bbox_xywh": None, "cutie_error": None, "cutie_ms": 0.0}
        try:
            if self.tracker is not None:
                t = time.monotonic_ns()
                try:
                    bbox = np.asarray(self.tracker.track(rgb), dtype=float)
                    if bbox.shape == (4,) and np.isfinite(bbox).all() and min(bbox[2:]) > 1:
                        self.diagnostic["bbox_xywh"] = bbox.tolist()
                        cx, cy = bbox[:2] + bbox[2:] / 2
                        if self.kf is None:
                            estimator.pose_last = adjust_pose_to_image_point(estimator.pose_last, K, cx, cy)
                        else:
                            pose_last = (
                                estimator.pose_last[0]
                                if estimator.pose_last.ndim == 3
                                else estimator.pose_last
                            )
                            z = float(pose_last[2, 3].detach().cpu())
                            if z > 1e-5:
                                self.mean, self.covariance = self.kf.update(
                                    self.mean,
                                    self.covariance,
                                    pose_to_6d(estimator.pose_last),
                                )
                                xy = np.array([(cx-K[0, 2])*z/K[0, 0], (cy-K[1, 2])*z/K[1, 1]])
                                self.mean, self.covariance = self.kf.update_from_xy(self.mean, self.covariance, xy)
                                estimator.pose_last = torch.as_tensor(sixd_to_pose(self.mean[:6]), device="cuda").unsqueeze(0)
                except Exception as exc:
                    self.diagnostic["cutie_error"] = str(exc)
                self.diagnostic["cutie_ms"] = (time.monotonic_ns()-t)/1e6
            raw = estimator.track_one(rgb, depth, K, iteration)
            if _finite_matrix(raw) is None:
                raise ValueError("nonfinite FoundationPose measurement")
            if self.kf is not None:
                # This is the original FP++ local sequence: update with the
                # previous pose only when Cutie has a bbox, run FoundationPose,
                # then predict one image step for the following frame. The raw
                # ``track_one`` result above is what the shared rig filter sees.
                self.mean, self.covariance = self.kf.predict(
                    self.mean, self.covariance
                )
            self.diagnostic["dt_s"] = self.dt
            return raw, (time.monotonic_ns()-started)/1e6, None
        except Exception as exc:
            estimator.pose_last = previous_pose
            self.mean, self.covariance = previous_mean, previous_cov
            return None, (time.monotonic_ns()-started)/1e6, str(exc)

    def set_timestamp(self, stamp, fps):
        dt = (stamp-self.stamp)*1e-9
        self.dt = dt if 0 < dt <= 0.25 else 1.0/fps
        self.stamp = stamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=Path("test/mesh/Box_53mm.STL"))
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument("--serial0", default="254322070766")
    parser.add_argument("--serial1", default="261722076377")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("../motion_tracking/config/calibration/current.yaml"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--est-refine-iter", type=int, default=10)
    parser.add_argument("--track-refine-iter", type=int, default=3)
    parser.add_argument("--activate-2d-tracker", action="store_true", help="Independent Cutie memory per camera, matching the single-view FP++ path.")
    parser.add_argument("--activate-kalman-filter", action="store_true", help="Local centered-mesh KF priors per camera; rig fusion uses raw FP outputs.")
    parser.add_argument(
        "--parallel-track",
        dest="parallel_track",
        action="store_true",
        default=False,
        help="Run both FoundationPose calls in worker threads (unsafe on some CUDA/nvdiffrast builds).",
    )
    parser.add_argument(
        "--serial-track",
        dest="parallel_track",
        action="store_false",
        help="Run camera trackers serially when a CUDA/nvdiffrast build is not thread-safe.",
    )
    parser.add_argument(
        "--use-fused-prior",
        dest="use_fused_prior",
        action="store_true",
        default=False,
        help="Reset both camera trackers to the shared fused rig prediction each frame.",
    )
    parser.add_argument(
        "--keep-local-prior",
        dest="use_fused_prior",
        action="store_false",
        help="Keep each camera's local pose_last between frames (default).",
    )
    parser.add_argument(
        "--max-track-translation-jump-mm",
        type=float,
        default=10.0,
        help="Reject a raw camera pose that moves farther than this in one processed frame (default: 10 mm).",
    )
    parser.add_argument(
        "--max-track-rotation-jump-deg",
        type=float,
        default=5.0,
        help="Reject a raw camera pose that rotates farther than this in one processed frame (default: 5 deg).",
    )
    parser.add_argument(
        "--max-track-speed-mm-s", type=float, default=1000.0,
        help="Maximum expected translation speed; jump gate scales with timestamp interval.",
    )
    parser.add_argument(
        "--max-track-rotation-speed-deg-s", type=float, default=360.0,
        help="Maximum expected rotation speed; jump gate scales with timestamp interval.",
    )
    parser.add_argument(
        "--max-skew-ms",
        type=float,
        default=50.0,
        help="Maximum host/device acquisition skew for a joint stereo update.",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=("host", "device"),
        default="host",
        help="Timestamp used for pairing (host is robust for unsynchronized USB clocks).",
    )
    parser.add_argument("--innovation-gate", type=float, default=16.812)
    parser.add_argument(
        "--reacquire-after-rejections",
        type=int,
        default=2,
        help="Reinitialize fusion after this many consecutive all-rejected stereo-consensus pairs.",
    )
    parser.add_argument(
        "--reacquire-translation-mm",
        type=float,
        default=20.0,
        help="Maximum translation disagreement for stereo-consensus reacquisition.",
    )
    parser.add_argument(
        "--reacquire-rotation-deg",
        type=float,
        default=15.0,
        help="Maximum rotation disagreement for stereo-consensus reacquisition.",
    )
    parser.add_argument("--roi0", type=parse_roi)
    parser.add_argument("--roi1", type=parse_roi)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("test/stereo_pose_debug.npz"))
    parser.add_argument(
        "--jsonl-output",
        type=str,
        default="test/stereo_pose_debug.jsonl",
        help="Per-pair JSONL diagnostics; pass an empty string to disable.",
    )
    args = parser.parse_args()
    plusplus = args.activate_2d_tracker or args.activate_kalman_filter
    if plusplus and (args.parallel_track or args.use_fused_prior):
        parser.error("FP++ requires --serial-track --keep-local-prior to keep independent local tracking states")
    if args.fps <= 0:
        parser.error("fps must be positive")
    if args.reacquire_after_rejections < 1:
        parser.error("--reacquire-after-rejections must be at least 1")
    if args.reacquire_translation_mm <= 0 or args.reacquire_rotation_deg <= 0:
        parser.error("reacquisition disagreement limits must be positive")
    args.jsonl_output = (
        None if not args.jsonl_output.strip() else Path(args.jsonl_output)
    )
    if not args.mesh.is_absolute():
        args.mesh = Path(__file__).resolve().parents[1] / args.mesh
    if not args.calibration.is_absolute():
        args.calibration = Path(__file__).resolve().parents[1] / args.calibration
    if args.no_display and (args.roi0 is None or args.roi1 is None):
        raise SystemExit("--no-display requires --roi0 and --roi1")
    if not args.mesh.exists():
        raise SystemExit(f"mesh not found: {args.mesh}")
    if not args.calibration.exists():
        raise SystemExit(f"calibration not found: {args.calibration}")

    repo_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "FoundationPose"))
    sys.path.insert(0, str(repo_dir / "src"))
    try:
        import pyrealsense2 as rs
        import torch
        from utils.stereo_pose_fusion import PoseMeasurement, StereoPoseFusion
    except ImportError as exc:
        raise SystemExit(f"missing pose dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by FoundationPose")

    t_rig_camera0, t_rig_camera1 = load_rig_extrinsics(args.calibration)
    logging.info("T_rig_camera_1=\n%s", t_rig_camera1)
    estimator0 = estimator1 = None
    pipelines = []
    records = []
    jsonl_stream = None
    track_executor = None
    reset_tracker0 = reset_tracker1 = False
    consensus_rejection_count = 0
    jsonl_records_since_flush = 0
    public0 = public1 = None
    last_observed_public0 = last_observed_public1 = None
    last_observed_stamp0 = last_observed_stamp1 = None
    pending_public0 = pending_public1 = None
    pending_count0 = pending_count1 = 0
    fusion = StereoPoseFusion(
        innovation_gate=args.innovation_gate,
        max_pair_skew_ns=int(args.max_skew_ms * 1e6),
    )
    try:
        estimator0, to_origin0, bbox0, draw_box, draw_axis = create_estimator(
            args.mesh, args.mesh_scale, repo_dir / "debug/stereo0", cube_symmetry=not plusplus
        )
        estimator1, to_origin1, bbox1, _, _ = create_estimator(
            args.mesh, args.mesh_scale, repo_dir / "debug/stereo1", cube_symmetry=not plusplus
        )
        pipeline0, align0, depth_scale0, K0, global_time0 = start_camera(
            rs, args.serial0, args.width, args.height, args.fps
        )
        pipelines.append(pipeline0)
        pipeline1, align1, depth_scale1, K1, global_time1 = start_camera(
            rs, args.serial1, args.width, args.height, args.fps
        )
        pipelines.append(pipeline1)
        logging.info("opened camera_0=%s and camera_1=%s", args.serial0, args.serial1)

        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="realsense")
        (first0, stamp0, _, first_frame_number0, device_stamp0), (first1, stamp1, _, first_frame_number1, device_stamp1) = wait_pair(
            executor,
            pipeline0,
            align0,
            args.timestamp_source == "device" and global_time0,
            pipeline1,
            align1,
            args.timestamp_source == "device" and global_time1,
            int(args.max_skew_ms * 1e6),
        )
        color0, depth0 = first0.get_color_frame(), first0.get_depth_frame()
        color1, depth1 = first1.get_color_frame(), first1.get_depth_frame()
        if not color0 or not depth0 or not color1 or not depth1:
            raise RuntimeError("both cameras must provide color and depth frames")
        bgr0 = np.asanyarray(color0.get_data())
        bgr1 = np.asanyarray(color1.get_data())
        roi0 = choose_roi("camera_0 FoundationPose", bgr0, args.roi0)
        roi1 = choose_roi("camera_1 FoundationPose", bgr1, args.roi1)
        rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
        rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
        depth_array0 = np.asanyarray(depth0.get_data()).astype(np.float32) * depth_scale0
        depth_array1 = np.asanyarray(depth1.get_data()).astype(np.float32) * depth_scale1
        public0, _ = register(estimator0, K0, rgb0, depth_array0, roi0, args.est_refine_iter)
        public1, _ = register(estimator1, K1, rgb1, depth_array1, roi1, args.est_refine_iter)
        # Registration may select a different one of the 24 equivalent cube
        # branches in the second view.  Resolve that branch before the first
        # fusion update, otherwise its corner-origin translation differs by a
        # cube diagonal and the camera is rejected forever.
        initial_rig_from_camera0 = t_rig_camera0 @ public0
        if estimator_is_cube(estimator1):
            public1 = canonicalize_cube_pose(
                public1,
                np.linalg.inv(t_rig_camera1) @ initial_rig_from_camera0,
                estimator1.model_center,
            )
        initial_measurements = [
            PoseMeasurement("camera_0", stamp0, public0, t_rig_camera0),
            PoseMeasurement("camera_1", stamp1, public1, t_rig_camera1),
        ]
        try:
            result = fusion.fuse(initial_measurements)
        except ValueError as exc:
            # Registration is still useful when the first two host waits were
            # not a close pair.  Initialize from camera_0 and let the next
            # valid pair perform the first stereo correction instead of
            # terminating before any overlay can be shown.
            logging.warning("initial stereo pair skipped: %s", exc)
            fusion.reset()
            result = fusion.fuse([initial_measurements[0]])
        local0 = local1 = None
        if plusplus:
            local0 = LocalPlusPlus(estimator0, rgb0, make_mask(rgb0.shape[:2], roi0), stamp0,
                                   args.activate_2d_tracker, args.activate_kalman_filter)
            local1 = LocalPlusPlus(estimator1, rgb1, make_mask(rgb1.shape[:2], roi1), stamp1,
                                   args.activate_2d_tracker, args.activate_kalman_filter)
        else:
            # Start both per-camera trackers from the same fused rig pose.  This
            # avoids carrying a camera-specific registration bias into the first
            # tracking iteration after cold start.
            estimator0.pose_last = public_to_internal(
                estimator0,
                np.linalg.inv(t_rig_camera0) @ result.t_rig_object,
                torch,
            )
            estimator1.pose_last = public_to_internal(
                estimator1,
                np.linalg.inv(t_rig_camera1) @ result.t_rig_object,
                torch,
            )
        # Raw jump validation is a per-camera continuity check.  Its reference
        # must advance independently of the shared fusion gate, otherwise one
        # fusion rejection freezes the camera's reference forever.
        last_observed_public0 = np.asarray(public0, dtype=np.float64).copy()
        last_observed_public1 = np.asarray(public1, dtype=np.float64).copy()
        last_observed_stamp0 = int(stamp0)
        last_observed_stamp1 = int(stamp1)
        logging.info("initial accepted=%s rejected=%s", result.accepted_sources, result.rejected_sources)
        if result.rejected_sources:
            logging.warning("initial stereo poses are inconsistent; check ROI and calibration")

        # FoundationPose emits a number of INFO messages in every call.  They
        # are useful while initializing, but terminal I/O in the hot loop can
        # add visible latency.  Keep the live loop quiet unless a call fails.
        logging.getLogger().setLevel(max(logging.getLogger().level, logging.WARNING))
        if args.parallel_track:
            track_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="foundationpose")
        if args.jsonl_output is not None:
            args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = args.jsonl_output.open("w", encoding="utf-8")

        while True:
            loop_start_ns = time.monotonic_ns()
            wait_start_ns = loop_start_ns
            (frames0, stamp0, host_stamp0, frame_number0, device_stamp0), (frames1, stamp1, host_stamp1, frame_number1, device_stamp1) = wait_pair(
                executor,
                pipeline0,
                align0,
                args.timestamp_source == "device" and global_time0,
                pipeline1,
                align1,
                args.timestamp_source == "device" and global_time1,
                int(args.max_skew_ms * 1e6),
            )
            wait_pair_ms = (time.monotonic_ns() - wait_start_ns) / 1e6
            color0, depth0 = frames0.get_color_frame(), frames0.get_depth_frame()
            color1, depth1 = frames1.get_color_frame(), frames1.get_depth_frame()
            if not color0 or not depth0 or not color1 or not depth1:
                continue
            bgr0 = np.asanyarray(color0.get_data())
            bgr1 = np.asanyarray(color1.get_data())
            rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
            rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
            depth_array0 = np.asanyarray(depth0.get_data()).astype(np.float32) * depth_scale0
            depth_array1 = np.asanyarray(depth1.get_data()).astype(np.float32) * depth_scale1

            # Save the state before prediction. If both measurements fail, we
            # restore this snapshot instead of allowing a stale velocity to
            # move the displayed object.
            fusion_snapshot = fusion.snapshot()
            # Advance the shared filter once to the oldest image in this pair.
            # Calling current_pose() here reused the previous frame's state as
            # the prior, which is especially harmful when the loop stalls.
            prediction_stamp = max(min(stamp0, stamp1), int(fusion.timestamp_ns or 0))
            prior_rig = fusion.predict_to(prediction_stamp)
            prior0 = prior1 = None
            if prior_rig is not None:
                prior0 = np.linalg.inv(t_rig_camera0) @ prior_rig
                prior1 = np.linalg.inv(t_rig_camera1) @ prior_rig
            if prior_rig is not None and not plusplus:
                # FoundationPose's local refiner is allowed to drift on a
                # textureless/symmetric cube.  Use the shared trusted rig
                # prediction as the prior for every frame; this bounds each
                # camera to the same physical track while the innovation and
                # jump gates below prevent one bad view from moving the rig.
                if estimator0.pose_last is None or reset_tracker0 or args.use_fused_prior:
                    estimator0.pose_last = public_to_internal(estimator0, prior0, torch)
                    reset_tracker0 = False
                if estimator1.pose_last is None or reset_tracker1 or args.use_fused_prior:
                    estimator1.pose_last = public_to_internal(estimator1, prior1, torch)
                    reset_tracker1 = False

            track_wall_start_ns = time.monotonic_ns()
            if plusplus:
                local0.set_timestamp(stamp0, args.fps)
                local1.set_timestamp(stamp1, args.fps)
                public0, camera0_track_ms, error0 = local0.track(estimator0, rgb0, depth_array0, K0, args.track_refine_iter)
                public1, camera1_track_ms, error1 = local1.track(estimator1, rgb1, depth_array1, K1, args.track_refine_iter)
            elif args.parallel_track:
                future0 = track_executor.submit(
                    track_one_timed, estimator0, rgb0, depth_array0, K0, args.track_refine_iter
                )
                future1 = track_executor.submit(
                    track_one_timed, estimator1, rgb1, depth_array1, K1, args.track_refine_iter
                )
                public0, camera0_track_ms, error0 = future0.result()
                public1, camera1_track_ms, error1 = future1.result()
            else:
                public0, camera0_track_ms, error0 = track_one_timed(
                    estimator0, rgb0, depth_array0, K0, args.track_refine_iter
                )
                public1, camera1_track_ms, error1 = track_one_timed(
                    estimator1, rgb1, depth_array1, K1, args.track_refine_iter
                )
            track_parallel_wall_ms = (time.monotonic_ns() - track_wall_start_ns) / 1e6
            if error0:
                logging.warning("camera_0 FoundationPose failed: %s", error0)
            if error1:
                logging.warning("camera_1 FoundationPose failed: %s", error1)
            # Keep the unmodified FoundationPose jump in the diagnostic log.
            # A cube-equivalent 180-degree branch can be corrected below, and
            # the before/after values make that correction auditable offline.
            uncanonicalized_jump0 = pose_jump_metrics(last_observed_public0, public0)
            uncanonicalized_jump1 = pose_jump_metrics(last_observed_public1, public1)
            # Maintain a continuous lift of the cube's 24-way symmetric pose.
            # Camera 0 follows its own previous raw observation; camera 1 is
            # aligned to camera 0's current branch.  A rejected shared-filter
            # prediction must not freeze either camera's symmetry reference.
            if estimator_is_cube(estimator0):
                public0 = canonicalize_cube_pose(
                    public0,
                    last_observed_public0
                    if last_observed_public0 is not None
                    else prior0,
                    estimator0.model_center,
                )
            if estimator_is_cube(estimator1):
                camera1_branch_reference = (
                    np.linalg.inv(t_rig_camera1) @ t_rig_camera0 @ public0
                    if public0 is not None
                    else last_observed_public1
                )
                public1 = canonicalize_cube_pose(
                    public1,
                    camera1_branch_reference
                    if camera1_branch_reference is not None
                    else prior1,
                    estimator1.model_center,
                )
            display_public0 = _finite_matrix(public0)
            display_public1 = _finite_matrix(public1)
            dt0 = min(0.25, max((int(stamp0) - int(last_observed_stamp0 or stamp0)) * 1e-9, 0.0))
            dt1 = min(0.25, max((int(stamp1) - int(last_observed_stamp1 or stamp1)) * 1e-9, 0.0))
            jump_limit_translation0 = max(float(args.max_track_translation_jump_mm) * 1e-3,
                                          float(args.max_track_speed_mm_s) * 1e-3 * dt0)
            jump_limit_translation1 = max(float(args.max_track_translation_jump_mm) * 1e-3,
                                          float(args.max_track_speed_mm_s) * 1e-3 * dt1)
            jump_limit_rotation0 = max(float(args.max_track_rotation_jump_deg),
                                       float(args.max_track_rotation_speed_deg_s) * dt0)
            jump_limit_rotation1 = max(float(args.max_track_rotation_jump_deg),
                                       float(args.max_track_rotation_speed_deg_s) * dt1)
            public0, pending_public0, pending_count0, raw_jump0, jump_reacquired0 = gate_pose_jump(
                public0, last_observed_public0, pending_public0, pending_count0,
                jump_limit_translation0, jump_limit_rotation0,
            )
            public1, pending_public1, pending_count1, raw_jump1, jump_reacquired1 = gate_pose_jump(
                public1, last_observed_public1, pending_public1, pending_count1,
                jump_limit_translation1, jump_limit_rotation1,
            )
            if display_public0 is not None and public0 is None:
                error0 = (
                    f"raw pose jump rejected: {raw_jump0[0] * 1000.0:.1f} mm, "
                    f"{raw_jump0[1]:.1f} deg"
                )
                logging.warning("camera_0 %s", error0)
            if display_public1 is not None and public1 is None:
                error1 = (
                    f"raw pose jump rejected: {raw_jump1[0] * 1000.0:.1f} mm, "
                    f"{raw_jump1[1]:.1f} deg"
                )
                logging.warning("camera_1 %s", error1)
            if jump_reacquired0:
                logging.warning("camera_0 raw jump gate relocked to a stable candidate")
            if jump_reacquired1:
                logging.warning("camera_1 raw jump gate relocked to a stable candidate")
            if public0 is not None:
                last_observed_public0 = np.asarray(public0, dtype=np.float64).copy()
                last_observed_stamp0 = int(stamp0)
            if public1 is not None:
                last_observed_public1 = np.asarray(public1, dtype=np.float64).copy()
                last_observed_stamp1 = int(stamp1)
            # Keep the estimator's internal centered-mesh state consistent
            # with the branch that survived canonicalization and the jump
            # gate.  This makes --keep-local-prior safe as well as the default
            # shared-prior mode.
            if public0 is not None and not plusplus:
                estimator0.pose_last = public_to_internal(estimator0, public0, torch)
            if public1 is not None and not plusplus:
                estimator1.pose_last = public_to_internal(estimator1, public1, torch)
            measurements = []
            if public0 is not None:
                measurements.append(PoseMeasurement("camera_0", stamp0, public0, t_rig_camera0))
            if public1 is not None:
                measurements.append(PoseMeasurement("camera_1", stamp1, public1, t_rig_camera1))
            raw_rig0 = (
                PoseMeasurement("camera_0", stamp0, public0, t_rig_camera0).in_rig().t_rig_object
                if public0 is not None else None
            )
            raw_rig1 = (
                PoseMeasurement("camera_1", stamp1, public1, t_rig_camera1).in_rig().t_rig_object
                if public1 is not None else None
            )
            fusion_start_ns = time.monotonic_ns()
            result = None
            fusion_error = None
            reacquired = False
            reacquisition_mode = None
            pre_reacquisition_innovation_distances = None
            consensus_translation_m, consensus_rotation_deg = pose_disagreement(
                raw_rig0, raw_rig1
            )
            try:
                if measurements:
                    result = fusion.fuse(measurements)
                else:
                    fusion_error = "both camera FoundationPose calls failed"
                    if fusion.initialized:
                        fusion.restore(fusion_snapshot)
                        fusion.hold_current()
                        result = fusion.result_from_state(
                            timestamp_ns=fusion_snapshot.timestamp_ns,
                            rejected_sources=("camera_0", "camera_1"),
                        )
            except ValueError as exc:
                # A late host pair is not a tracking failure.  Drop this pair
                # so the shared filter never receives a time-regressing or
                # over-skewed update, then continue with the next frameset.
                fusion_error = str(exc)
                logging.warning("stereo pair skipped (skew=%.1f ms): %s", abs(stamp1 - stamp0) / 1e6, exc)
                # Prediction happened before the pair was validated.  A pair
                # that cannot be fused must not advance the displayed state or
                # leave a velocity that will extrapolate through later drops.
                if fusion.initialized:
                    fusion.restore(fusion_snapshot)
                    fusion.hold_current()
                    result = fusion.result_from_state(
                        timestamp_ns=fusion_snapshot.timestamp_ns,
                        rejected_sources=tuple(m.source for m in measurements),
                    )
            # A sudden hand rotation can put both valid measurements outside
            # an over-confident temporal gate.  Holding the old state forever
            # then makes every subsequent innovation still larger.  Two
            # independent cameras that agree with each other are stronger
            # evidence than that stale prediction, so use consecutive stereo
            # consensus to reinitialize the shared state.
            if result is not None and not result.accepted_sources:
                pre_reacquisition_innovation_distances = dict(
                    result.innovation_distances
                )
                stereo_consensus = (
                    fusion_error is None
                    and len(measurements) == 2
                    and consensus_translation_m
                    <= float(args.reacquire_translation_mm) * 1e-3
                    and consensus_rotation_deg <= float(args.reacquire_rotation_deg)
                )
                consensus_rejection_count = (
                    consensus_rejection_count + 1 if stereo_consensus else 0
                )
                if consensus_rejection_count >= args.reacquire_after_rejections:
                    fusion.reset()
                    result = fusion.fuse(measurements)
                    reacquired = True
                    reacquisition_mode = "stereo_consensus"
                    consensus_rejection_count = 0
                    logging.warning(
                        "fusion reacquired from stereo consensus (%.1f mm, %.1f deg)",
                        consensus_translation_m * 1000.0,
                        consensus_rotation_deg,
                    )
                elif not stereo_consensus:
                    # The per-camera jump gate has already removed implausible
                    # raw poses. If the cameras disagree, retain availability
                    # by anchoring to the measurement closest to the old
                    # prediction rather than freezing forever. The other
                    # camera remains rejected and can rejoin normally later.
                    candidates = [
                        (
                            result.innovation_distances.get(
                                measurement.source, float("inf")
                            ),
                            measurement,
                        )
                        for measurement in measurements
                    ]
                    distance, candidate = min(
                        candidates,
                        default=(float("inf"), None),
                        key=lambda item: item[0],
                    )
                    if candidate is not None and np.isfinite(distance):
                        fusion.reset()
                        result = fusion.fuse([candidate])
                        result.rejected_sources = [
                            measurement.source
                            for measurement in measurements
                            if measurement.source != candidate.source
                        ]
                        reacquired = True
                        reacquisition_mode = "best_camera"
                        logging.warning(
                            "fusion reacquired from %s (previous innovation %.2f)",
                            candidate.source,
                            distance,
                        )
                    else:
                        fusion.restore(fusion_snapshot)
                        fusion.hold_current()
                        result = fusion.result_from_state(
                            timestamp_ns=fusion_snapshot.timestamp_ns,
                            rejected_sources=[m.source for m in measurements],
                            innovation_distances=result.innovation_distances,
                        )
                else:
                    fusion.restore(fusion_snapshot)
                    fusion.hold_current()
                    result = fusion.result_from_state(
                        timestamp_ns=fusion_snapshot.timestamp_ns,
                        rejected_sources=[m.source for m in measurements],
                        innovation_distances=result.innovation_distances,
                    )
            else:
                consensus_rejection_count = 0
            fusion_ms = (time.monotonic_ns() - fusion_start_ns) / 1e6
            skew_ms = abs(stamp1 - stamp0) / 1e6
            draw_start_ns = time.monotonic_ns()
            record = {
                "index": len(records),
                "local_plusplus": {"camera_0": local0.diagnostic, "camera_1": local1.diagnostic} if plusplus else None,
                "camera_0": {
                    "frame_number": int(frame_number0),
                    "capture_timestamp_ns": int(stamp0),
                    "host_capture_timestamp_ns": int(host_stamp0),
                    "device_timestamp_ns": int(device_stamp0),
                    "t_camera_object": _json_matrix(public0),
                    "candidate_t_camera_object": _json_matrix(display_public0),
                    "t_rig_object": _json_matrix(raw_rig0),
                },
                "camera_1": {
                    "frame_number": int(frame_number1),
                    "capture_timestamp_ns": int(stamp1),
                    "host_capture_timestamp_ns": int(host_stamp1),
                    "device_timestamp_ns": int(device_stamp1),
                    "t_camera_object": _json_matrix(public1),
                    "candidate_t_camera_object": _json_matrix(display_public1),
                    "t_rig_object": _json_matrix(raw_rig1),
                },
                "pair_skew_ms": float(skew_ms),
                "fused": {
                    "timestamp_ns": int(result.timestamp_ns) if result is not None else None,
                    "t_rig_object": _json_matrix(result.t_rig_object) if result is not None else None,
                    "accepted_sources": list(result.accepted_sources) if result is not None else [],
                    "rejected_sources": list(result.rejected_sources) if result is not None else [],
                },
                "innovation_distance": {
                    "camera_0": _json_float(
                        (pre_reacquisition_innovation_distances or result.innovation_distances).get(
                            "camera_0", np.nan
                        )
                    ) if result is not None else None,
                    "camera_1": _json_float(
                        (pre_reacquisition_innovation_distances or result.innovation_distances).get(
                            "camera_1", np.nan
                        )
                    ) if result is not None else None,
                },
                "timing": {
                    "wait_pair_ms": float(wait_pair_ms),
                    "track_camera_0_ms": float(camera0_track_ms),
                    "track_camera_1_ms": float(camera1_track_ms),
                    "track_parallel_wall_ms": float(track_parallel_wall_ms),
                    "fusion_ms": float(fusion_ms),
                    "draw_ms": 0.0,
                    "loop_total_ms": 0.0,
                    "display_timestamp_ns": None,
                    "display_age_ms": None,
                },
                "fusion_error": fusion_error,
                "reacquisition": {
                    "reacquired": bool(reacquired),
                    "mode": reacquisition_mode,
                    "consensus_rejection_count": int(consensus_rejection_count),
                    "camera_disagreement_translation_mm": _json_float(
                        consensus_translation_m * 1000.0
                    ),
                    "camera_disagreement_rotation_deg": _json_float(
                        consensus_rotation_deg
                    ),
                },
                "tracking_error": {"camera_0": error0, "camera_1": error1},
                "jump_gate_reacquired": {
                    "camera_0": bool(jump_reacquired0),
                    "camera_1": bool(jump_reacquired1),
                },
                "raw_jump": {
                    "camera_0_translation_mm": _json_float(raw_jump0[0] * 1000.0),
                    "camera_0_rotation_deg": _json_float(raw_jump0[1]),
                    "camera_1_translation_mm": _json_float(raw_jump1[0] * 1000.0),
                    "camera_1_rotation_deg": _json_float(raw_jump1[1]),
                },
                "uncanonicalized_jump": {
                    "camera_0_translation_mm": _json_float(uncanonicalized_jump0[0] * 1000.0),
                    "camera_0_rotation_deg": _json_float(uncanonicalized_jump0[1]),
                    "camera_1_translation_mm": _json_float(uncanonicalized_jump1[0] * 1000.0),
                    "camera_1_rotation_deg": _json_float(uncanonicalized_jump1[1]),
                },
                "tracking_mode": {
                    "parallel": bool(args.parallel_track),
                    "use_fused_prior": bool(args.use_fused_prior),
                    "max_translation_jump_mm": float(args.max_track_translation_jump_mm),
                    "max_rotation_jump_deg": float(args.max_track_rotation_jump_deg),
                    "max_speed_mm_s": float(args.max_track_speed_mm_s),
                    "max_rotation_speed_deg_s": float(args.max_track_rotation_speed_deg_s),
                    "jump_limit_camera_0": [jump_limit_translation0 * 1000.0, jump_limit_rotation0],
                    "jump_limit_camera_1": [jump_limit_translation1 * 1000.0, jump_limit_rotation1],
                },
            }
            if result is not None:
                reset_tracker0 = "camera_0" in result.rejected_sources
                reset_tracker1 = "camera_1" in result.rejected_sources
            reset_tracker0 = reset_tracker0 or error0 is not None
            reset_tracker1 = reset_tracker1 or error1 is not None
            if not args.no_display:
                # Green is the fused pose.  Blue/red are the raw camera
                # measurements, making it immediately visible which source
                # introduces a lag or disagreement.
                overlay0 = bgr0.copy()
                overlay1 = bgr1.copy()
                if display_public0 is not None:
                    # draw_posed_3d_box receives RGB colors; convert the
                    # intended display colors accordingly before the final
                    # RGB->BGR conversion.
                    overlay0 = draw_pose_overlay(overlay0, display_public0, to_origin0, bbox0, K0, draw_box, draw_axis, "camera_0 raw", (0, 0, 255), False)
                if display_public1 is not None:
                    overlay1 = draw_pose_overlay(overlay1, display_public1, to_origin1, bbox1, K1, draw_box, draw_axis, "camera_1 raw", (255, 0, 0), False)
                if result is not None:
                    camera_pose0 = np.linalg.inv(t_rig_camera0) @ result.t_rig_object
                    camera_pose1 = np.linalg.inv(t_rig_camera1) @ result.t_rig_object
                    overlay0 = draw_pose_overlay(overlay0, camera_pose0, to_origin0, bbox0, K0, draw_box, draw_axis, "fused", (0, 255, 0) if result.accepted_sources else (255, 165, 0), True)
                    overlay1 = draw_pose_overlay(overlay1, camera_pose1, to_origin1, bbox1, K1, draw_box, draw_axis, "fused", (0, 255, 0) if result.accepted_sources else (255, 165, 0), True)
                status = (
                    f"{'MEASURED' if result is not None and result.accepted_sources else 'STALE'} | raw blue/red | "
                    f"age={(time.monotonic_ns() - max(host_stamp0, host_stamp1)) / 1e6:.0f}ms "
                    f"loop={(time.monotonic_ns() - loop_start_ns) / 1e6:.0f}ms "
                    f"track={track_parallel_wall_ms:.0f}ms"
                )
                for overlay in (overlay0, overlay1):
                    cv2.putText(
                        overlay, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA,
                    )
                display_timestamp_ns = time.monotonic_ns()
                display_age_ms = (display_timestamp_ns - max(host_stamp0, host_stamp1)) / 1e6
                record["timing"]["display_timestamp_ns"] = int(display_timestamp_ns)
                record["timing"]["display_age_ms"] = float(display_age_ms)
                cv2.imshow("camera_0 FoundationPose", overlay0)
                cv2.imshow("camera_1 FoundationPose", overlay1)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    draw_ms = (time.monotonic_ns() - draw_start_ns) / 1e6
                    record["timing"]["draw_ms"] = float(draw_ms)
                    record["timing"]["loop_total_ms"] = float((time.monotonic_ns() - loop_start_ns) / 1e6)
                    display_timestamp_ns = record["timing"]["display_timestamp_ns"]
                    if display_timestamp_ns is None:
                        display_timestamp_ns = time.monotonic_ns()
                    if jsonl_stream is not None:
                        jsonl_stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                        jsonl_stream.flush()
                    records.append(record)
                    break
            draw_ms = (time.monotonic_ns() - draw_start_ns) / 1e6
            record["timing"]["draw_ms"] = float(draw_ms)
            record["timing"]["loop_total_ms"] = float((time.monotonic_ns() - loop_start_ns) / 1e6)
            records.append(record)
            if jsonl_stream is not None:
                jsonl_stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                jsonl_records_since_flush += 1
                # Keep the log useful if the process is interrupted without
                # forcing an fsync-like flush on every high-rate frame.
                if jsonl_records_since_flush >= 8:
                    jsonl_stream.flush()
                    jsonl_records_since_flush = 0
    finally:
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        if track_executor is not None:
            track_executor.shutdown(wait=True, cancel_futures=True)
        if jsonl_stream is not None:
            jsonl_stream.close()
        if "executor" in locals():
            executor.shutdown(wait=True, cancel_futures=True)
        if not args.no_display:
            cv2.destroyAllWindows()
        if records:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            count = len(records)
            nan_pose = np.full((count, 4, 4), np.nan, dtype=np.float64)
            camera0_pose = nan_pose.copy()
            camera1_pose = nan_pose.copy()
            camera0_rig = nan_pose.copy()
            camera1_rig = nan_pose.copy()
            fused_rig = nan_pose.copy()
            timestamps = np.full((count, 2), -1, dtype=np.int64)
            host_timestamps = np.full((count, 2), -1, dtype=np.int64)
            device_timestamps = np.full((count, 2), -1, dtype=np.int64)
            display_timestamps = np.full(count, -1, dtype=np.int64)
            frame_numbers = np.full((count, 2), -1, dtype=np.int64)
            skew_values = np.full(count, np.nan, dtype=np.float64)
            innovations = np.full((count, 2), np.nan, dtype=np.float64)
            timing_keys = (
                "wait_pair_ms", "track_camera_0_ms", "track_camera_1_ms",
                "track_parallel_wall_ms", "fusion_ms", "draw_ms",
                "loop_total_ms", "display_age_ms",
            )
            timing_values = {key: np.full(count, np.nan, dtype=np.float64) for key in timing_keys}
            accepted_sources = np.empty(count, dtype=object)
            rejected_sources = np.empty(count, dtype=object)
            for index, item in enumerate(records):
                camera0_pose[index] = np.asarray(item["camera_0"]["t_camera_object"], dtype=np.float64) if item["camera_0"]["t_camera_object"] is not None else nan_pose[index]
                camera1_pose[index] = np.asarray(item["camera_1"]["t_camera_object"], dtype=np.float64) if item["camera_1"]["t_camera_object"] is not None else nan_pose[index]
                camera0_rig[index] = np.asarray(item["camera_0"]["t_rig_object"], dtype=np.float64) if item["camera_0"]["t_rig_object"] is not None else nan_pose[index]
                camera1_rig[index] = np.asarray(item["camera_1"]["t_rig_object"], dtype=np.float64) if item["camera_1"]["t_rig_object"] is not None else nan_pose[index]
                if item["fused"]["t_rig_object"] is not None:
                    fused_rig[index] = np.asarray(item["fused"]["t_rig_object"], dtype=np.float64)
                timestamps[index] = [item["camera_0"]["capture_timestamp_ns"], item["camera_1"]["capture_timestamp_ns"]]
                host_timestamps[index] = [item["camera_0"]["host_capture_timestamp_ns"], item["camera_1"]["host_capture_timestamp_ns"]]
                device_timestamps[index] = [item["camera_0"]["device_timestamp_ns"], item["camera_1"]["device_timestamp_ns"]]
                if item["timing"].get("display_timestamp_ns") is not None:
                    display_timestamps[index] = item["timing"]["display_timestamp_ns"]
                frame_numbers[index] = [item["camera_0"]["frame_number"], item["camera_1"]["frame_number"]]
                skew_values[index] = item["pair_skew_ms"]
                for source_index, source in enumerate(("camera_0", "camera_1")):
                    value = item["innovation_distance"][source]
                    if value is not None:
                        innovations[index, source_index] = value
                for key in timing_keys:
                    value = item["timing"].get(key)
                    if value is not None:
                        timing_values[key][index] = value
                accepted_sources[index] = tuple(item["fused"]["accepted_sources"])
                rejected_sources[index] = tuple(item["fused"]["rejected_sources"])
            np.savez_compressed(
                args.output,
                records=np.asarray(records, dtype=object),
                frame_numbers=frame_numbers,
                capture_timestamps_ns=timestamps,
                host_capture_timestamps_ns=host_timestamps,
                device_timestamps_ns=device_timestamps,
                display_timestamps_ns=display_timestamps,
                pair_skew_ms=skew_values,
                camera_0_t_camera_object=camera0_pose,
                camera_1_t_camera_object=camera1_pose,
                camera_0_t_rig_object=camera0_rig,
                camera_1_t_rig_object=camera1_rig,
                fused_t_rig_object=fused_rig,
                innovation_distances=innovations,
                accepted_sources=accepted_sources,
                rejected_sources=rejected_sources,
                **timing_values,
                calibration=str(args.calibration),
                serial0=args.serial0,
                serial1=args.serial1,
                jsonl_output=str(args.jsonl_output) if args.jsonl_output is not None else "",
            )
            logging.info("saved %d stereo poses to %s", len(records), args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
