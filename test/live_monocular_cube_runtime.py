#!/usr/bin/env python3
"""Camera-0 FoundationPose++ cube tracking with camera-1 projection and DR-NAS IPC.

AprilTags initialize (or re-initialize) FoundationPose; they are never fused
into the normal tracking output.  The second camera is visualization-only.
"""

from __future__ import annotations

import argparse
import errno
import logging
import math
import signal
import socket
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve()
FP_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[2]
MOTION_ROOT = REPO_ROOT / "motion_tracking"
EXPORTER_SOURCE = (
    MOTION_ROOT / "ros2_ws" / "src" / "realsense_pose_exporter"
)
for item in (
    SCRIPT.parent, FP_ROOT, FP_ROOT / "FoundationPose", FP_ROOT / "src",
    EXPORTER_SOURCE,
):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from live_stereo_realsense_pose import (  # noqa: E402
    LocalPlusPlus,
    canonicalize_cube_pose,
    create_estimator,
    pose_jump_metrics,
    pose_to_6d,
)
from realsense_pose_exporter.protocol import (  # noqa: E402
    boot_id,
    encode_packet,
    monotonic_timens_offset_ns,
)


CUBE_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)
LOGGER = logging.getLogger("foundationpose_monocular_runtime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cube-config", type=Path,
        default=MOTION_ROOT / "config/objects/cube_53p11.yaml",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=MOTION_ROOT / "config/calibration/current.yaml",
    )
    parser.add_argument(
        "--locked-reference", type=Path,
        default=MOTION_ROOT / "config/camera_tags/wuji_palm_id1_locked.yaml",
    )
    parser.add_argument(
        "--mesh", type=Path, default=FP_ROOT / "test/mesh/Box_53mm.STL"
    )
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument(
        "--rgb-width", type=int, default=None,
        help="RGB capture width; defaults to the calibration stream width.",
    )
    parser.add_argument(
        "--rgb-height", type=int, default=None,
        help="RGB capture height; defaults to the calibration stream height.",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="RGB/depth capture FPS; defaults to the calibration stream FPS.",
    )
    parser.add_argument(
        "--intrinsics-source", choices=("auto", "calibration", "realsense"),
        default="auto",
        help=("auto uses calibrated K/D at the calibrated RGB size and active "
              "RealSense profile intrinsics at any overridden RGB size."),
    )
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=720)
    parser.add_argument("--est-refine-iter", type=int, default=10)
    parser.add_argument("--track-refine-iter", type=int, default=3)
    parser.add_argument(
        "--inference-scale", type=float, default=0.5,
        help="Camera-0 RGB-D scale used by Cutie/FoundationPose (display stays at capture resolution).",
    )
    parser.add_argument("--socket-path", type=Path,
                        default=MOTION_ROOT / "runtime/tracked_objects.pose.sock")
    parser.add_argument("--exposure-us", type=float, default=2500.0)
    parser.add_argument("--gain", type=float, default=80.0)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--no-2d-tracker", action="store_true")
    parser.add_argument("--no-kalman-filter", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--display-width", type=int, default=960,
        help="Initial width of each camera pane; the window remains resizable.",
    )
    parser.add_argument(
        "--scene-height", type=int, default=0,
        help=("Initial height of the rig 3-D pane; 0 matches the initial "
              "camera-pane height for an equal top/bottom split."),
    )
    parser.add_argument(
        "--scene-frame", choices=("locked-tag", "world"), default="locked-tag",
        help=("3-D display frame. locked-tag uses the locked reference Tag "
              "plane as the table and its outward +Z normal as up; world "
              "retains the raw rig/calibration orientation."),
    )
    parser.add_argument("--tag-check-period", type=int, default=3)
    parser.add_argument("--tag-reprojection-p95-px", type=float, default=4.0)
    parser.add_argument("--reinit-translation-mm", type=float, default=55.0)
    parser.add_argument("--reinit-rotation-deg", type=float, default=50.0)
    parser.add_argument("--reinit-confirmations", type=int, default=2)
    parser.add_argument("--max-invalid-frames", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument(
        "--frame-timeout-ms", type=int, default=2000,
        help="How long camera_0 may stop delivering frames before both streams are restarted.",
    )
    parser.add_argument(
        "--max-camera-restarts", type=int, default=3,
        help="Maximum automatic RealSense stream restarts before exiting.",
    )
    args = parser.parse_args()
    if args.tag_check_period < 1 or args.reinit_confirmations < 1:
        parser.error("tag periods and confirmation counts must be positive")
    if (args.max_invalid_frames < 1 or args.display_width < 160 or
            args.scene_height < 0 or 0 < args.scene_height < 180):
        parser.error("invalid frame limit/display/scene size")
    if (args.rgb_width is None) != (args.rgb_height is None):
        parser.error("--rgb-width and --rgb-height must be provided together")
    if (args.rgb_width is not None and
            (args.rgb_width < 1 or args.rgb_height < 1)):
        parser.error("RGB dimensions must be positive")
    if args.fps is not None and args.fps < 1:
        parser.error("--fps must be positive")
    if args.depth_width < 1 or args.depth_height < 1:
        parser.error("depth dimensions must be positive")
    if args.frame_timeout_ms < 250 or args.max_camera_restarts < 0:
        parser.error("frame timeout must be >= 250 ms and restart count must be non-negative")
    if not 0.1 <= args.inference_scale <= 1.0:
        parser.error("--inference-scale must be in [0.1, 1.0]")
    if not args.auto_exposure and (args.exposure_us <= 0 or args.gain < 0):
        parser.error("manual exposure and gain are invalid")
    return args


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"file is not readable: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def transform_from_yaml(value: dict) -> np.ndarray:
    q = np.asarray(value["quaternion_wxyz"], dtype=np.float64)
    t = np.asarray(value["translation_xyz_m"], dtype=np.float64)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(np.r_[q, t]).all():
        raise ValueError("transform contains invalid quaternion/translation")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("transform quaternion is zero")
    q /= norm
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    output[:3, 3] = t
    return output


@dataclass(frozen=True)
class RigConfig:
    calibration_id: str
    width: int
    height: int
    fps: int
    serial0: str
    serial1: str
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    t_rig_camera0: np.ndarray
    t_rig_camera1: np.ndarray


def load_rig(path: Path) -> RigConfig:
    doc = _read_yaml(path)
    cameras = doc.get("cameras")
    stream = doc.get("stream")
    if not isinstance(cameras, list) or len(cameras) != 2 or not isinstance(stream, dict):
        raise ValueError("calibration must contain exactly two cameras and a stream")
    by_name = {item.get("logical_name"): item for item in cameras}
    c0, c1 = by_name["camera_0"], by_name["camera_1"]
    return RigConfig(
        calibration_id=str(doc["calibration_id"]),
        width=int(stream["width"]), height=int(stream["height"]),
        fps=int(stream["fps"]), serial0=str(c0["serial"]), serial1=str(c1["serial"]),
        K0=np.asarray(c0["K"], dtype=np.float64).reshape(3, 3),
        D0=np.asarray(c0["D"], dtype=np.float64),
        K1=np.asarray(c1["K"], dtype=np.float64).reshape(3, 3),
        D1=np.asarray(c1["D"], dtype=np.float64),
        t_rig_camera0=transform_from_yaml(doc["T_rig_camera_0"]),
        t_rig_camera1=transform_from_yaml(doc["T_rig_camera_1"]),
    )


@dataclass(frozen=True)
class CubeModel:
    name: str
    tag_size_m: float
    object_points: dict[int, np.ndarray]
    half_extent_m: float

    @property
    def corners(self) -> np.ndarray:
        h = self.half_extent_m
        return np.asarray([
            [-h, -h, -h], [h, -h, -h], [-h, h, -h], [h, h, -h],
            [-h, -h, h], [h, -h, h], [-h, h, h], [h, h, h],
        ], dtype=np.float64)


def load_cube(path: Path) -> CubeModel:
    doc = _read_yaml(path)
    if doc.get("tag_family") != "tag36h11":
        raise ValueError("cube tag_family must be tag36h11")
    size = float(doc["tag_size_m"])
    half = size * 0.5
    # AprilTag/Kalibr convention: lower-left, lower-right, upper-right,
    # upper-left, with +X printed-right and +Y printed-up.
    tag_corners = np.asarray([
        [-half, -half, 0.0], [half, -half, 0.0],
        [half, half, 0.0], [-half, half, 0.0],
    ], dtype=np.float64)
    points: dict[int, np.ndarray] = {}
    face_offsets = []
    for entry in doc["tags"]:
        tag_id = int(entry["id"])
        t_tag_object = transform_from_yaml(entry["T_tag_object"])
        points[tag_id] = (
            (tag_corners - t_tag_object[:3, 3]) @ t_tag_object[:3, :3]
        )
        face_offsets.append(float(np.linalg.norm(t_tag_object[:3, 3])))
    if not points or len(points) != len(doc["tags"]):
        raise ValueError("cube config contains no tags or duplicate IDs")
    return CubeModel(str(doc["object_name"]), size, points,
                     float(np.median(face_offsets)))


@dataclass(frozen=True)
class LockedPose:
    name: str
    transform: np.ndarray
    tag_id: int
    tag_transform: np.ndarray
    covariance: list[float]


def load_locked(path: Path, calibration_id: str) -> LockedPose:
    doc = _read_yaml(path)
    if doc.get("frame_id") != "world" or doc.get("rig_calibration_id") != calibration_id:
        raise ValueError("locked reference frame/calibration does not match current calibration")
    transform = transform_from_yaml(doc["pose_world_object"])
    tag_transform = transform_from_yaml(doc["pose_world_tag"])
    translation_std = max(float(doc.get("position_rms_mm", 1.0)) * 1e-3, 1e-5)
    rotation_std = max(math.radians(float(doc.get("rotation_rms_deg", 0.2))), 1e-5)
    covariance = np.diag([translation_std**2] * 3 + [rotation_std**2] * 3)
    return LockedPose(str(doc["object_name"]), transform, int(doc["tag_id"]),
                      tag_transform,
                      covariance.reshape(-1).tolist())


class CubeTagDetector:
    def __init__(self, model: CubeModel, max_p95_px: float) -> None:
        self.model = model
        self.max_p95_px = float(max_p95_px)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.raw_ids: list[int] = []
        self.raw_corners = []

    def detect(self, bgr: np.ndarray, K: np.ndarray, D: np.ndarray):
        corners, ids, _ = self.detector.detectMarkers(bgr)
        self.raw_corners = corners
        self.raw_ids = [] if ids is None else [int(value) for value in ids.reshape(-1)]
        if ids is None:
            return None, [], float("inf")
        selected = []
        for image_corners, raw_id in zip(corners, ids.reshape(-1)):
            tag_id = int(raw_id)
            if tag_id in self.model.object_points:
                # OpenCV gives upper-left, upper-right, lower-right,
                # lower-left. Reorder to the repository's AprilTag convention.
                selected.append((tag_id, image_corners.reshape(4, 2)[[3, 2, 1, 0]]))
        if not selected:
            return None, [], float("inf")
        object_points = np.concatenate(
            [self.model.object_points[tag_id] for tag_id, _ in selected]
        ).astype(np.float64)
        image_points = np.concatenate([value for _, value in selected]).astype(np.float64)
        try:
            if len(selected) == 1:
                ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    object_points, image_points, K, D, flags=cv2.SOLVEPNP_IPPE
                )
                if not ok or not rvecs:
                    return None, [], float("inf")
                candidates = []
                for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    error = float(np.asarray(errors).reshape(-1)[index])
                    rotation, _ = cv2.Rodrigues(rvec)
                    translation = np.asarray(tvec).reshape(3)
                    points_camera = object_points @ rotation.T + translation
                    face = self.model.object_points[selected[0][0]]
                    center_object = face.mean(axis=0)
                    normal_object = np.cross(face[1] - face[0], face[3] - face[0])
                    normal_object /= np.linalg.norm(normal_object)
                    center_camera = rotation @ center_object + translation
                    normal_camera = rotation @ normal_object
                    outward_visible = float(normal_camera @ (-center_camera)) > 0.0
                    if np.all(points_camera[:, 2] > 0.0) and outward_visible:
                        candidates.append((error, rvec, tvec))
                if not candidates:
                    return None, [], float("inf")
                _, rvec, tvec = min(candidates, key=lambda value: value[0])
            else:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points, image_points, K, D, iterationsCount=100,
                    reprojectionError=self.max_p95_px, confidence=0.999,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if not ok or inliers is None or len(inliers) < 4:
                    return None, [], float("inf")
                cv2.solvePnP(
                    object_points[inliers.reshape(-1)], image_points[inliers.reshape(-1)],
                    K, D, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE,
                )
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
        except cv2.error:
            return None, [], float("inf")
        residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        p95 = float(np.percentile(residuals, 95))
        if not np.isfinite(p95) or p95 > self.max_p95_px:
            return None, [item[0] for item in selected], p95
        rotation, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = np.asarray(tvec).reshape(3)
        if pose[2, 3] <= 0:
            return None, [item[0] for item in selected], p95
        return pose, sorted(item[0] for item in selected), p95


def project_points(points: np.ndarray, pose: np.ndarray, K: np.ndarray,
                   D: np.ndarray) -> np.ndarray | None:
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    if np.any(camera_points[:, 2] <= 1e-4):
        return None
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    pixels, _ = cv2.projectPoints(points, rvec, pose[:3, 3], K, D)
    return pixels.reshape(-1, 2)


def initialization_mask(shape: tuple[int, int], cube: CubeModel,
                        pose: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray | None:
    pixels = project_points(cube.corners, pose, K, D)
    if pixels is None:
        return None
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    pad = max(5, round(0.06 * math.sqrt(max(cv2.contourArea(hull), 1.0))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    return cv2.dilate(mask, kernel)


def draw_cube(image: np.ndarray, cube: CubeModel, pose: np.ndarray | None,
              K: np.ndarray, D: np.ndarray, color: tuple[int, int, int],
              label: str) -> np.ndarray:
    output = image.copy()
    if pose is None:
        return output
    pixels = project_points(cube.corners, pose, K, D)
    if pixels is None:
        return output
    pixels = np.rint(pixels).astype(int)
    for left, right in CUBE_EDGES:
        cv2.line(output, tuple(pixels[left]), tuple(pixels[right]), color, 3, cv2.LINE_AA)
    origin = project_points(np.zeros((1, 3)), pose, K, D)
    if origin is not None:
        cv2.putText(output, label, tuple(np.rint(origin[0]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return output


def draw_roi(image: np.ndarray, mask: np.ndarray | None,
             label: str = "AUTO ROI from cube tags") -> np.ndarray:
    output = image.copy()
    if mask is None or not np.any(mask):
        return output
    selected = mask > 0
    tint = np.zeros_like(output)
    tint[:, :, 2] = 255
    output[selected] = cv2.addWeighted(
        output[selected], 0.70, tint[selected], 0.30, 0.0
    )
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 0, 255), 3, cv2.LINE_AA)
    if contours:
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.putText(output, label, (x, max(24, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2,
                    cv2.LINE_AA)
    return output


def draw_foundationpose_overlay(
    bgr: np.ndarray,
    mesh_pose: np.ndarray | None,
    semantic_pose: np.ndarray | None,
    to_origin: np.ndarray,
    bbox: np.ndarray,
    K: np.ndarray,
    draw_posed_3d_box,
    draw_xyz_axis,
    label: str,
    line_color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Render the same centered mesh box and XYZ axes as FoundationPose."""
    if mesh_pose is None:
        return bgr.copy()
    center_pose = mesh_pose @ np.linalg.inv(to_origin)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = draw_posed_3d_box(
        K, img=rgb, ob_in_cam=center_pose, bbox=bbox,
        line_color=line_color_rgb, linewidth=3,
    )
    rgb = draw_xyz_axis(
        rgb, ob_in_cam=center_pose, scale=0.04, K=K,
        thickness=3, transparency=0, is_input_rgb=True,
    )
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if semantic_pose is not None:
        xyz = semantic_pose[:3, 3]
        cv2.putText(
            output,
            f"{label} t=[{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}] m",
            (18, output.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.60, line_color_rgb[::-1], 2, cv2.LINE_AA,
        )
    return output


def pose_is_plausible(pose: np.ndarray | None, cube: CubeModel, depth: np.ndarray,
                      K: np.ndarray, D: np.ndarray, min_depth: float,
                      max_depth: float) -> tuple[bool, str]:
    if pose is None or np.asarray(pose).shape != (4, 4) or not np.isfinite(pose).all():
        return False, "nonfinite_pose"
    z = float(pose[2, 3])
    if not min_depth <= z <= max_depth:
        return False, "pose_depth_out_of_range"
    pixels = project_points(cube.corners, pose, K, D)
    if pixels is None:
        return False, "cube_behind_camera"
    height, width = depth.shape
    if (pixels[:, 0].max() < 0 or pixels[:, 0].min() >= width or
            pixels[:, 1].max() < 0 or pixels[:, 1].min() >= height):
        return False, "projection_outside_image"
    mask = initialization_mask(depth.shape, cube, pose, K, D)
    values = depth[(mask > 0) & (depth > 1e-4)] if mask is not None else np.empty(0)
    if values.size < 30:
        return False, "insufficient_object_depth"
    # The mask includes all cube faces and a small margin. Use a generous
    # consistency gate only to recognize gross drift, not to refine the pose.
    if abs(float(np.median(values)) - z) > 0.14:
        return False, "object_depth_disagrees"
    return True, "tracked"


def reset_local(local: LocalPlusPlus | None, estimator, rgb: np.ndarray,
                mask: np.ndarray, stamp_ns: int, use_2d: bool, use_kf: bool):
    if local is None:
        return LocalPlusPlus(estimator, rgb, mask, stamp_ns, use_2d, use_kf)
    if local.tracker is not None:
        local.tracker.cutie_processor.clear_memory()
        local.tracker.initialize(rgb, init_info={"mask": (mask > 0).astype(np.uint8)})
    if local.kf is not None:
        local.mean, local.covariance = local.kf.initiate(pose_to_6d(estimator.pose_last))
    local.stamp = stamp_ns
    return local


def seed_estimator_public_pose(estimator, public_pose: np.ndarray) -> None:
    """Set FoundationPose's centered internal pose from its public mesh pose."""
    if estimator.pose_last is None:
        raise RuntimeError("FoundationPose has no registered pose to reseed")
    tf_to_center = estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
    centered_pose = np.asarray(public_pose, dtype=np.float64) @ np.linalg.inv(tf_to_center)
    estimator.pose_last = estimator.pose_last.new_tensor(centered_pose)


def system_monotonic_offset() -> tuple[int, int]:
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return wall - (before + after) // 2, after - before


def frame_timestamp(rs, frame) -> tuple[int, int, int] | None:
    domain = frame.get_frame_timestamp_domain()
    if domain not in (rs.timestamp_domain.global_time, rs.timestamp_domain.system_time):
        return None
    try:
        frame_us = float(frame.get_frame_metadata(rs.frame_metadata_value.frame_timestamp))
        sensor_us = float(frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp))
    except RuntimeError:
        return None
    stamp_ms = float(frame.get_timestamp()) + (sensor_us - frame_us) / 1000.0
    if not np.isfinite(stamp_ms) or stamp_ms <= 0:
        return None
    ros_ns = int(round(stamp_ms * 1e6))
    offset, bracket_ns = system_monotonic_offset()
    mono_ns = ros_ns - offset
    uncertainty_ns = 500_000 + bracket_ns // 2
    now_ns = time.monotonic_ns()
    if mono_ns <= 0 or mono_ns > now_ns + 5_000_000 or uncertainty_ns > 2_000_000:
        return None
    return ros_ns, mono_ns, uncertainty_ns


def configure_color_sensor(rs, profile, auto_exposure: bool,
                           exposure_us: float, gain: float) -> None:
    for sensor in profile.get_device().query_sensors():
        try:
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)
            is_color_sensor = any(
                stream.stream_type() == rs.stream.color
                for stream in sensor.get_stream_profiles()
            )
            if not is_color_sensor:
                continue
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            if not auto_exposure and sensor.supports(rs.option.exposure):
                # D435i RGB exposure option uses 100-us units.
                sensor.set_option(rs.option.exposure, float(exposure_us) / 100.0)
            if not auto_exposure and sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain, float(gain))
        except RuntimeError as exc:
            LOGGER.warning("camera option was not applied: %s", exc)


def active_color_intrinsics(rs, profile) -> tuple[np.ndarray, np.ndarray, str]:
    """Convert the active RealSense color profile to OpenCV K/D."""
    video = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = video.get_intrinsics()
    if intrinsics.model not in {
            rs.distortion.none, rs.distortion.brown_conrady,
            rs.distortion.modified_brown_conrady,
            rs.distortion.inverse_brown_conrady}:
        raise RuntimeError(
            f"active color distortion model {intrinsics.model} cannot be "
            "used directly by OpenCV projectPoints/solvePnP"
        )
    K = np.asarray([
        [intrinsics.fx, 0.0, intrinsics.ppx],
        [0.0, intrinsics.fy, intrinsics.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    if intrinsics.model == rs.distortion.none:
        return K, np.zeros(5, dtype=np.float64), str(intrinsics.model)
    if intrinsics.model in {
            rs.distortion.brown_conrady,
            rs.distortion.modified_brown_conrady}:
        return (K, np.asarray(intrinsics.coeffs[:5], dtype=np.float64),
                str(intrinsics.model))

    # D435i color profiles may expose inverse Brown-Conrady coefficients:
    # those map image pixels to ideal rays and cannot be passed to OpenCV as
    # forward distortion. Sample that exact librealsense deprojection and fit
    # the equivalent forward OpenCV k1,k2,p1,p2,k3 model.
    rows = []
    targets = []
    samples = []
    for pixel_y in np.linspace(0.0, intrinsics.height - 1.0, 17):
        for pixel_x in np.linspace(0.0, intrinsics.width - 1.0, 25):
            point = np.asarray(rs.rs2_deproject_pixel_to_point(
                intrinsics, [float(pixel_x), float(pixel_y)], 1.0
            ), dtype=np.float64)
            x, y = point[0] / point[2], point[1] / point[2]
            xd = (pixel_x - intrinsics.ppx) / intrinsics.fx
            yd = (pixel_y - intrinsics.ppy) / intrinsics.fy
            r2 = x * x + y * y
            r4, r6 = r2 * r2, r2 * r2 * r2
            rows.extend((
                [x * r2, x * r4, 2.0 * x * y,
                 r2 + 2.0 * x * x, x * r6],
                [y * r2, y * r4, r2 + 2.0 * y * y,
                 2.0 * x * y, y * r6],
            ))
            targets.extend((xd - x, yd - y))
            samples.append((x, y, pixel_x, pixel_y))
    D, _, _, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64), rcond=None,
    )
    ideal = np.asarray([[x, y, 1.0] for x, y, _, _ in samples],
                       dtype=np.float64)
    projected, _ = cv2.projectPoints(
        ideal, np.zeros(3), np.zeros(3), K, D
    )
    expected = np.asarray([[u, v] for _, _, u, v in samples], dtype=np.float64)
    errors = np.linalg.norm(projected.reshape(-1, 2) - expected, axis=1)
    p95, maximum = float(np.percentile(errors, 95)), float(errors.max())
    if not np.isfinite(errors).all() or maximum > 0.75:
        raise RuntimeError(
            "could not approximate inverse Brown-Conrady as an OpenCV forward "
            f"model (p95={p95:.3f}px max={maximum:.3f}px)"
        )
    model = (f"{intrinsics.model}->opencv_brown_fit"
             f"(p95={p95:.3f}px,max={maximum:.3f}px)")
    return K, D, model


def start_cameras(rs, rig: RigConfig, args: argparse.Namespace):
    rgb_width = rig.width if args.rgb_width is None else args.rgb_width
    rgb_height = rig.height if args.rgb_height is None else args.rgb_height
    fps = rig.fps if args.fps is None else args.fps
    calibrated_size = (rgb_width, rgb_height) == (rig.width, rig.height)
    if args.intrinsics_source == "calibration" and not calibrated_size:
        raise ValueError(
            "--intrinsics-source calibration requires the calibrated RGB size "
            f"{rig.width}x{rig.height}; requested {rgb_width}x{rgb_height}"
        )
    pipe0, config0 = rs.pipeline(), rs.config()
    config0.enable_device(rig.serial0)
    config0.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)
    config0.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, fps)
    pipe1 = None
    started0 = False
    started1 = False
    try:
        profile0 = pipe0.start(config0)
        started0 = True
        configure_color_sensor(
            rs, profile0, args.auto_exposure, args.exposure_us, args.gain
        )
        pipe1, config1 = rs.pipeline(), rs.config()
        config1.enable_device(rig.serial1)
        config1.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)
        profile1 = pipe1.start(config1)
        started1 = True
        configure_color_sensor(rs, profile1, args.auto_exposure, args.exposure_us, args.gain)
        depth_scale = profile0.get_device().first_depth_sensor().get_depth_scale()
        use_calibration = (args.intrinsics_source == "calibration" or
                           (args.intrinsics_source == "auto" and calibrated_size))
        if use_calibration:
            K0, D0, K1, D1 = rig.K0.copy(), rig.D0.copy(), rig.K1.copy(), rig.D1.copy()
            source = "external_calibration"
        else:
            K0, D0, model0 = active_color_intrinsics(rs, profile0)
            K1, D1, model1 = active_color_intrinsics(rs, profile1)
            source = f"realsense_active_profile(camera0={model0},camera1={model1})"
        return (pipe0, rs.align(rs.stream.color), depth_scale, pipe1,
                K0, D0, K1, D1, rgb_width, rgb_height, fps, source)
    except BaseException:
        if started1:
            try:
                pipe1.stop()
            except Exception:
                pass
        if started0:
            try:
                pipe0.stop()
            except Exception:
                pass
        raise


class CameraFrameTimeout(RuntimeError):
    """The pose-source stream stopped producing frames."""


def newest_frames(pipeline, camera_name: str, timeout_ms: int):
    try:
        frames = pipeline.wait_for_frames(timeout_ms)
    except RuntimeError as exc:
        message = str(exc)
        raise CameraFrameTimeout(f"{camera_name}: {message}") from exc
    while True:
        try:
            newer = pipeline.poll_for_frames()
        except RuntimeError:
            return frames
        if not newer:
            return frames
        frames = newer


def poll_newest_frames(pipeline):
    """Return the newest immediately available frameset without blocking."""
    frames = None
    while True:
        try:
            newer = pipeline.poll_for_frames()
        except RuntimeError:
            return None
        if not newer:
            return frames
        frames = newer


class DatagramPublisher:
    def __init__(self, path: Path, calibration_id: str,
                 object_names: tuple[str, str]) -> None:
        encoded = str(path).encode()
        if not path.is_absolute() or len(encoded) >= 104:
            raise ValueError("socket path must be absolute and shorter than 104 bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.calibration_id = calibration_id
        self.stream_id = str(uuid.uuid4())
        self.boot = boot_id()
        self.timens_offset = monotonic_timens_offset_ns()
        self.packet_seq = 0
        self.measurement_seq = {name: 0 for name in object_names}
        self.state = {name: "INITIALIZING" for name in object_names}
        self.last_heartbeat_ns = {name: 0 for name in object_names}
        self.absent_drops = 0
        self.receiver_present = False

    def close(self) -> None:
        self.socket.close()

    def _base(self, name: str, event: str, reason: str) -> dict:
        self.packet_seq += 1
        return {
            "event": event, "stream_id": self.stream_id, "packet_seq": self.packet_seq,
            "measurement_seq": self.measurement_seq[name],
            "producer_state": self.state[name] if event == "HEARTBEAT" else event,
            "boot_id": self.boot, "monotonic_timens_offset_ns": self.timens_offset,
            "object_name": name, "rig_calibration_id": self.calibration_id,
            "world_frame": "world", "source_mask": 0,
            "timestamp_authoritative": False, "timestamp_quality": "unavailable",
            "acquisition_ros_ns": 0, "acquisition_monotonic_ns": 0,
            "producer_send_monotonic_ns": time.monotonic_ns(),
            "timestamp_uncertainty_ns": 0, "reason": reason,
            "position_world_m": None, "quaternion_world_xyzw": None,
            "twist_world": None, "pose_covariance": None, "twist_covariance": None,
            "innovation_chi2": None, "estimated_skew_ms": None,
            "conservative_skew_ms": None, "camera_frame_numbers": [0, 0],
            "visible_tag_ids": [], "visibility_masks": [],
        }

    def _send(self, packet: dict) -> bool:
        packet["producer_send_monotonic_ns"] = time.monotonic_ns()
        try:
            self.socket.sendto(encode_packet(packet), self.path)
            self.receiver_present = True
            return True
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ECONNREFUSED):
                self.absent_drops += 1
                self.receiver_present = False
                return False
            if exc.errno in (errno.EAGAIN, errno.ENOBUFS):
                return False
            raise

    def measured(self, name: str, pose: np.ndarray, covariance: list[float],
                 twist: list[float], timestamp: tuple[int, int, int],
                 frame_number: int, visible_ids: list[int], reason: str) -> None:
        self.measurement_seq[name] += 1
        self.state[name] = "MEASURED"
        ros_ns, mono_ns, uncertainty_ns = timestamp
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
        packet = self._base(name, "MEASURED", reason)
        packet.update({
            "source_mask": 1, "timestamp_authoritative": True,
            "timestamp_quality": "exposure_midpoint_global",
            "acquisition_ros_ns": ros_ns, "acquisition_monotonic_ns": mono_ns,
            "timestamp_uncertainty_ns": uncertainty_ns,
            "position_world_m": pose[:3, 3].astype(float).tolist(),
            "quaternion_world_xyzw": quaternion.astype(float).tolist(),
            "twist_world": [float(value) for value in twist],
            "pose_covariance": covariance,
            "twist_covariance": np.diag([0.25**2] * 3 + [2.0**2] * 3).reshape(-1).tolist(),
            "innovation_chi2": 0.0, "estimated_skew_ms": 0.0,
            "conservative_skew_ms": uncertainty_ns * 2e-6,
            "camera_frame_numbers": [int(frame_number), 0],
            "visible_tag_ids": sorted(int(value) for value in visible_ids),
            "visibility_masks": [1] * len(visible_ids),
        })
        self._send(packet)

    def lost(self, name: str, reason: str) -> None:
        if self.state[name] == "LOST":
            return
        self.state[name] = "LOST"
        self._send(self._base(name, "LOST", reason))

    def heartbeat(self) -> None:
        now = time.monotonic_ns()
        for name in self.state:
            if now - self.last_heartbeat_ns[name] >= 250_000_000:
                self.last_heartbeat_ns[name] = now
                self._send(self._base(name, "HEARTBEAT", f"producer_alive_state_{self.state[name].lower()}"))


def twist_between(previous: np.ndarray | None, current: np.ndarray,
                  previous_ns: int | None, current_ns: int) -> list[float]:
    if previous is None or previous_ns is None:
        return [0.0] * 6
    dt = (current_ns - previous_ns) * 1e-9
    if not 0.005 <= dt <= 0.5:
        return [0.0] * 6
    linear = (current[:3, 3] - previous[:3, 3]) / dt
    angular = Rotation.from_matrix(current[:3, :3] @ previous[:3, :3].T).as_rotvec() / dt
    return np.r_[linear, angular].astype(float).tolist()


@dataclass
class SceneViewState:
    """Mutable orbit-camera state shared with the OpenCV mouse callback."""

    yaw: float = 0.0
    pitch: float = math.radians(35.0)
    zoom: float = 1.0
    default_yaw: float = 0.0
    default_pitch: float = math.radians(35.0)
    pan_x: float = 0.0
    pan_y: float = 0.0
    dragging: bool = False
    panning: bool = False
    last_x: int = 0
    last_y: int = 0
    scene_top: int = 0
    scene_width: int = 1
    scene_height: int = 1

    def set_default_from_eye(self, eye_direction: np.ndarray) -> None:
        eye = np.asarray(eye_direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(eye))
        if norm < 1e-8:
            eye = np.asarray([1.0, -1.0, 1.0], dtype=np.float64)
            norm = float(np.linalg.norm(eye))
        eye /= norm
        self.default_yaw = math.atan2(float(eye[1]), float(eye[0]))
        self.default_pitch = math.asin(float(np.clip(eye[2], -1.0, 1.0)))
        self.reset()

    def reset(self) -> None:
        self.yaw = self.default_yaw
        self.pitch = self.default_pitch
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.dragging = False
        self.panning = False

    def eye_direction(self) -> np.ndarray:
        horizontal = math.cos(self.pitch)
        return np.asarray([
            horizontal * math.cos(self.yaw),
            horizontal * math.sin(self.yaw),
            math.sin(self.pitch),
        ], dtype=np.float64)


def scene_mouse_callback(event: int, x: int, y: int, flags: int,
                         state: SceneViewState) -> None:
    """Orbit/zoom only when the pointer is over the bottom 3-D pane."""
    in_scene = y >= state.scene_top
    if event == cv2.EVENT_LBUTTONDOWN and in_scene:
        state.dragging = True
        state.panning = False
        state.last_x, state.last_y = x, y
    elif event == cv2.EVENT_MBUTTONDOWN and in_scene:
        state.panning = True
        state.dragging = False
        state.last_x, state.last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
        dx, dy = x - state.last_x, y - state.last_y
        state.yaw -= dx * 0.008
        state.pitch = float(np.clip(
            state.pitch + dy * 0.008,
            math.radians(5.0), math.radians(88.0),
        ))
        state.last_x, state.last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and state.panning:
        dx, dy = x - state.last_x, y - state.last_y
        state.pan_x += dx / max(state.scene_width, 1)
        state.pan_y += dy / max(state.scene_height, 1)
        state.pan_x = float(np.clip(state.pan_x, -1.5, 1.5))
        state.pan_y = float(np.clip(state.pan_y, -1.5, 1.5))
        state.last_x, state.last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        state.dragging = False
    elif event == cv2.EVENT_MBUTTONUP:
        state.panning = False
    elif event == cv2.EVENT_MOUSEWHEEL and in_scene:
        if hasattr(cv2, "getMouseWheelDelta"):
            delta = int(cv2.getMouseWheelDelta(flags))
        else:
            delta = (int(flags) >> 16) & 0xFFFF
            if delta >= 0x8000:
                delta -= 0x10000
        if delta:
            state.zoom = float(np.clip(
                state.zoom * math.pow(1.12, delta / 120.0), 0.35, 5.0
            ))


class TrackingWindow:
    """Resizable Tk canvas whose wheel is not intercepted by Qt HighGUI."""

    def __init__(self, title: str, width: int, height: int,
                 scene_view: SceneViewState) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        self.scene_view = scene_view
        self.closed = False
        self._keys: list[int] = []
        self._photo = None
        self._image_item = None
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(320, 240)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas = tk.Canvas(
            self.root, background="#181b20", highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<KeyPress>", self._on_key)
        self.canvas.bind("<ButtonPress-1>", self._left_down)
        self.canvas.bind("<B1-Motion>", self._left_move)
        self.canvas.bind("<ButtonRelease-1>", self._left_up)
        self.canvas.bind("<ButtonPress-2>", self._middle_down)
        self.canvas.bind("<B2-Motion>", self._middle_move)
        self.canvas.bind("<ButtonRelease-2>", self._middle_up)
        self.canvas.bind("<MouseWheel>", self._wheel)
        # X11/Tk reports vertical wheel steps as buttons 4 and 5.
        self.canvas.bind("<Button-4>", self._wheel)
        self.canvas.bind("<Button-5>", self._wheel)
        self.canvas.focus_set()
        self.root.update_idletasks()
        self.root.update()

    def _on_close(self) -> None:
        self.closed = True
        try:
            self.root.withdraw()
        except self._tk.TclError:
            pass

    def _on_key(self, event) -> None:
        if event.keysym == "Escape":
            self._keys.append(27)
        elif event.char:
            self._keys.append(ord(event.char[0]))

    def _mouse(self, cv_event: int, event, flags: int = 0) -> None:
        scene_mouse_callback(cv_event, int(event.x), int(event.y),
                             flags, self.scene_view)

    def _left_down(self, event) -> None:
        self._mouse(cv2.EVENT_LBUTTONDOWN, event, cv2.EVENT_FLAG_LBUTTON)

    def _left_move(self, event) -> None:
        self._mouse(cv2.EVENT_MOUSEMOVE, event, cv2.EVENT_FLAG_LBUTTON)

    def _left_up(self, event) -> None:
        self._mouse(cv2.EVENT_LBUTTONUP, event)

    def _middle_down(self, event) -> None:
        self._mouse(cv2.EVENT_MBUTTONDOWN, event, cv2.EVENT_FLAG_MBUTTON)

    def _middle_move(self, event) -> None:
        self._mouse(cv2.EVENT_MOUSEMOVE, event, cv2.EVENT_FLAG_MBUTTON)

    def _middle_up(self, event) -> None:
        self._mouse(cv2.EVENT_MBUTTONUP, event)

    def _wheel(self, event) -> str:
        if getattr(event, "num", None) == 4:
            delta = 120
        elif getattr(event, "num", None) == 5:
            delta = -120
        else:
            delta = int(getattr(event, "delta", 0))
        flags = (delta & 0xFFFF) << 16
        self._mouse(cv2.EVENT_MOUSEWHEEL, event, flags)
        return "break"

    def size(self) -> tuple[int, int]:
        if self.closed:
            return 320, 240
        self.root.update_idletasks()
        return (max(320, int(self.canvas.winfo_width())),
                max(240, int(self.canvas.winfo_height())))

    def show(self, bgr: np.ndarray) -> None:
        if self.closed:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(rgb)
        self._photo = self._ImageTk.PhotoImage(image=image)
        if self._image_item is None:
            self._image_item = self.canvas.create_image(
                0, 0, anchor=self._tk.NW, image=self._photo,
            )
        else:
            self.canvas.itemconfigure(self._image_item, image=self._photo)
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:
            self.closed = True

    def poll_key(self) -> int:
        return self._keys.pop(0) if self._keys else -1

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass


def combined_view(left: np.ndarray, right: np.ndarray, width: int,
                  top_height: int, scene: np.ndarray | None = None) -> np.ndarray:
    def resize_to_pane(image: np.ndarray, pane_width: int,
                       pane_height: int) -> np.ndarray:
        canvas = np.zeros((pane_height, pane_width, 3), dtype=np.uint8)
        scale = min(pane_width / image.shape[1], pane_height / image.shape[0])
        target = (max(1, round(image.shape[1] * scale)),
                  max(1, round(image.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, target, interpolation=interpolation)
        x0 = (pane_width - target[0]) // 2
        y0 = (pane_height - target[1]) // 2
        canvas[y0:y0 + target[1], x0:x0 + target[0]] = resized
        return canvas

    left_width = width // 2
    right_width = width - left_width
    top = np.hstack((resize_to_pane(left, left_width, top_height),
                     resize_to_pane(right, right_width, top_height)))
    if scene is None:
        return top
    if scene.shape[1] != width:
        scene = cv2.resize(
            scene, (width, scene.shape[0]), interpolation=cv2.INTER_AREA
        )
    return np.vstack((top, scene))


def render_rig_scene(
    width: int,
    height: int,
    rig: RigConfig,
    locked: LockedPose,
    cube: CubeModel,
    cube_world: np.ndarray | None,
    cube_state: str,
    K0: np.ndarray,
    K1: np.ndarray,
    rgb_width: int,
    rgb_height: int,
    scene_frame: str = "locked-tag",
    view_state: SceneViewState | None = None,
) -> np.ndarray:
    """Render an RViz-like view without changing any tracking/IPC frames."""
    canvas = np.full((height, width, 3), (24, 27, 32), dtype=np.uint8)
    margin = 24
    scene_left = margin
    scene_right = width - margin
    scene_top = margin
    scene_bottom = height - margin
    if scene_frame == "locked-tag":
        # The object definition uses AprilTag/Kalibr axes: tag +Z points out
        # from the printed face. Since this tag lies parallel to the table,
        # T_tag_world makes its face the scene XY plane and its normal scene +Z.
        scene_from_world = np.linalg.inv(locked.tag_transform)
        display_frame_label = f"table/tag_{locked.tag_id} (+Z up)"
        bounds_xyz = ((-0.35, 0.35), (-0.25, 0.65), (-0.12, 0.55))
        grid_x = (-0.35, 0.35)
        grid_y = (-0.25, 0.65)
    else:
        scene_from_world = np.eye(4, dtype=np.float64)
        display_frame_label = "world/rig"
        bounds_xyz = ((-0.22, 0.68), (-0.42, 0.28), (-0.08, 0.72))
        grid_x = (-0.3, 0.8)
        grid_y = (-0.6, 0.4)
    # Stable fixed workspace prevents the view from zooming as the cube moves.
    bounds = np.asarray([
        [x, y, z]
        for x in bounds_xyz[0]
        for y in bounds_xyz[1]
        for z in bounds_xyz[2]
    ], dtype=np.float64)
    default_eye = (scene_from_world @ rig.t_rig_camera0)[:3, 3]
    if float(np.linalg.norm(default_eye)) < 1e-8:
        default_eye = np.asarray([1.0, -1.0, 1.0], dtype=np.float64)
    default_eye /= np.linalg.norm(default_eye)
    eye_direction = (default_eye if view_state is None
                     else view_state.eye_direction())
    forward = -eye_direction
    screen_right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(screen_right, forward)
    screen_up /= np.linalg.norm(screen_up)
    bound_u, bound_v = bounds @ screen_right, bounds @ screen_up
    scale = min(
        (scene_right - scene_left) / max(float(np.ptp(bound_u)), 1e-6),
        (scene_bottom - scene_top) / max(float(np.ptp(bound_v)), 1e-6),
    ) * (1.12 if view_state is None else 1.12 * view_state.zoom)
    center_u = 0.5 * (float(bound_u.min()) + float(bound_u.max()))
    center_v = 0.5 * (float(bound_v.min()) + float(bound_v.max()))

    def project(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        u, v = values @ screen_right, values @ screen_up
        return np.column_stack((
            width * (0.52 + (0.0 if view_state is None else view_state.pan_x))
            + (u - center_u) * scale,
            (scene_top + scene_bottom) * 0.5
            + height * (0.0 if view_state is None else view_state.pan_y)
            - (v - center_v) * scale,
        )).round().astype(np.int32)

    def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return points @ pose[:3, :3].T + pose[:3, 3]

    def rpy_degrees(pose: np.ndarray) -> np.ndarray:
        rotation = pose[:3, :3]
        horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
        if horizontal > 1e-8:
            roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            pitch = math.atan2(-float(rotation[2, 0]), horizontal)
            yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        else:
            roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
            pitch = math.atan2(-float(rotation[2, 0]), horizontal)
            yaw = 0.0
        return np.degrees([roll, pitch, yaw])

    def line3(a, b, color, thickness=1, arrow=False):
        pixels = project(np.asarray([a, b]))
        if arrow:
            cv2.arrowedLine(canvas, tuple(pixels[0]), tuple(pixels[1]), color,
                            thickness, cv2.LINE_AA, tipLength=0.16)
        else:
            cv2.line(canvas, tuple(pixels[0]), tuple(pixels[1]), color,
                     thickness, cv2.LINE_AA)

    def label3(point, text, color=(230, 230, 230), offset=(6, -6), scale_text=0.46):
        pixel = project(np.asarray(point).reshape(1, 3))[0]
        origin = (int(pixel[0] + offset[0]), int(pixel[1] + offset[1]))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale_text, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale_text, color, 1, cv2.LINE_AA)

    def axes(pose: np.ndarray, length: float, label: str,
             label_color=(230, 230, 230), thickness=2,
             label_offset=(6, -6)):
        local = np.asarray([
            [0.0, 0.0, 0.0], [length, 0.0, 0.0],
            [0.0, length, 0.0], [0.0, 0.0, length],
        ])
        world = transform_points(pose, local)
        # RViz convention: X red, Y green, Z blue (OpenCV BGR below).
        line3(world[0], world[1], (0, 0, 255), thickness, True)
        line3(world[0], world[2], (0, 255, 0), thickness, True)
        line3(world[0], world[3], (255, 80, 40), thickness, True)
        label3(world[3], label, label_color, label_offset)

    def camera_frustum(pose: np.ndarray, K: np.ndarray, name: str,
                       color: tuple[int, int, int]):
        depth = 0.12
        half_x = depth * rgb_width * 0.5 / max(float(K[0, 0]), 1e-6)
        half_y = depth * rgb_height * 0.5 / max(float(K[1, 1]), 1e-6)
        local = np.asarray([
            [0.0, 0.0, 0.0],
            [-half_x, -half_y, depth], [half_x, -half_y, depth],
            [half_x, half_y, depth], [-half_x, half_y, depth],
        ])
        world = transform_points(pose, local)
        for index in range(1, 5):
            line3(world[0], world[index], color, 2)
            line3(world[index], world[1 + index % 4], color, 2)
        offset = (-86, 22) if name == "camera_0" else (8, -8)
        axes(pose, 0.075, name, color, 2, offset)

    # The grid is the locked Tag/table plane in locked-tag mode.
    for value in np.arange(math.ceil(grid_x[0] * 10.0) / 10.0,
                           grid_x[1] + 0.001, 0.1):
        major = abs(value) < 1e-8
        color = (80, 84, 91) if major else (48, 52, 58)
        line3([value, grid_y[0], 0.0], [value, grid_y[1], 0.0],
              color, 2 if major else 1)
    for value in np.arange(math.ceil(grid_y[0] * 10.0) / 10.0,
                           grid_y[1] + 0.001, 0.1):
        major = abs(value) < 1e-8
        color = (80, 84, 91) if major else (48, 52, 58)
        line3([grid_x[0], value, 0.0], [grid_x[1], value, 0.0],
              color, 2 if major else 1)

    camera0_scene = scene_from_world @ rig.t_rig_camera0
    camera1_scene = scene_from_world @ rig.t_rig_camera1
    palm_scene = scene_from_world @ locked.transform
    cube_scene = None if cube_world is None else scene_from_world @ cube_world
    world_scene = scene_from_world

    axes(np.eye(4), 0.14, display_frame_label, (255, 255, 255), 3, (8, 35))
    if scene_frame == "locked-tag":
        axes(world_scene, 0.10, "world/rig", (175, 175, 175), 2, (7, 19))
    camera_frustum(camera0_scene, K0, "camera_0", (255, 180, 50))
    camera_frustum(camera1_scene, K1, "camera_1", (255, 120, 220))
    line3(camera0_scene[:3, 3], camera1_scene[:3, 3],
          (150, 150, 150), 2)

    # Locked palm platform: RViz-like axes, gold base outline, and label.
    palm_plate = np.asarray([
        [-0.06, -0.04, 0.0], [0.06, -0.04, 0.0],
        [0.06, 0.04, 0.0], [-0.06, 0.04, 0.0],
    ])
    palm_world = transform_points(palm_scene, palm_plate)
    for index in range(4):
        line3(palm_world[index], palm_world[(index + 1) % 4], (0, 205, 255), 3)
    axes(palm_scene, 0.12, f"{locked.name}_locked", (0, 220, 255),
         3, (-92, -10))

    cube_color = (60, 235, 80) if cube_world is not None else (100, 100, 100)
    if cube_world is not None:
        corners_world = transform_points(cube_scene, cube.corners)
        for left, right in CUBE_EDGES:
            line3(corners_world[left], corners_world[right], cube_color, 3)
        axes(cube_scene, 0.085, "cube_53p11", cube_color, 3, (8, -9))

    camera_relative = np.linalg.inv(rig.t_rig_camera0) @ rig.t_rig_camera1
    baseline = float(np.linalg.norm(camera_relative[:3, 3]))
    camera_rpy = rpy_degrees(camera_relative)
    palm_rpy = rpy_degrees(locked.transform)
    lines = [
        (f"3-D DISPLAY: {display_frame_label}  "
         "(X=red, Y=green, Z=blue; grid=0.10 m)"),
        ("view: left-drag orbit | middle-drag pan | wheel zoom "
         f"({1.0 if view_state is None else view_state.zoom:.2f}x) | V/0 reset"),
        "numeric poses below retain original world/relative semantics",
        ("camera_1 in camera_0: "
         f"t=[{camera_relative[0,3]:+.3f}, {camera_relative[1,3]:+.3f}, "
         f"{camera_relative[2,3]:+.3f}] m  baseline={baseline:.3f} m"),
        ("  camera_1 rpy="
         f"[{camera_rpy[0]:+.1f}, {camera_rpy[1]:+.1f}, "
         f"{camera_rpy[2]:+.1f}] deg"),
        (f"{locked.name}_locked in world: "
         f"t=[{locked.transform[0,3]:+.3f}, {locked.transform[1,3]:+.3f}, "
         f"{locked.transform[2,3]:+.3f}] m"),
        (f"  {locked.name}_locked rpy="
         f"[{palm_rpy[0]:+.1f}, {palm_rpy[1]:+.1f}, "
         f"{palm_rpy[2]:+.1f}] deg"),
    ]
    if cube_world is None:
        lines.append(f"cube_53p11: LOST / INITIALIZING  ({cube_state})")
    else:
        palm_cube = np.linalg.inv(locked.transform) @ cube_world
        cube_world_rpy = rpy_degrees(cube_world)
        palm_cube_rpy = rpy_degrees(palm_cube)
        lines.extend((
            ("cube in world: "
             f"t=[{cube_world[0,3]:+.3f}, {cube_world[1,3]:+.3f}, "
             f"{cube_world[2,3]:+.3f}] m"),
            ("  cube world rpy="
             f"[{cube_world_rpy[0]:+.1f}, {cube_world_rpy[1]:+.1f}, "
             f"{cube_world_rpy[2]:+.1f}] deg"),
            (f"cube relative to {locked.name}: "
             f"t=[{palm_cube[0,3]:+.3f}, {palm_cube[1,3]:+.3f}, "
             f"{palm_cube[2,3]:+.3f}] m"),
            (f"  cube/{locked.name} rpy="
             f"[{palm_cube_rpy[0]:+.1f}, {palm_cube_rpy[1]:+.1f}, "
             f"{palm_cube_rpy[2]:+.1f}] deg"),
            f"cube state: {cube_state[:64]}",
        ))
    overlay = canvas.copy()
    text_panel_right = min(width - 4, max(700, int(round(width * 0.48))))
    text_panel_bottom = min(height - 4, 30 + len(lines) * 20)
    cv2.rectangle(overlay, (4, 3), (text_panel_right, text_panel_bottom),
                  (12, 14, 18), cv2.FILLED)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0.0, canvas)
    for index, text_line in enumerate(lines):
        y = 22 + index * 20
        color = ((90, 245, 110) if cube_world is not None and index == len(lines) - 1
                 else (80, 100, 255) if cube_world is None and index == len(lines) - 1
                 else (225, 225, 225))
        cv2.putText(canvas, text_line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text_line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, color, 1, cv2.LINE_AA)

    controls = (
        "3-D CONTROLS",
        "LMB drag  : orbit",
        "MMB drag  : pan",
        "Wheel     : zoom 3-D only",
        "V / 0     : reset 3-D view",
        "R         : re-register cube",
        "Q / Esc   : quit",
    )
    control_width = 286
    control_height = 18 + len(controls) * 20
    control_left = max(4, width - control_width - 8)
    control_right = width - 4
    control_bottom = min(height - 4, 3 + control_height)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (control_left, 3),
                  (control_right, control_bottom), (12, 14, 18), cv2.FILLED)
    cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0.0, canvas)
    for index, text_line in enumerate(controls):
        y = 22 + index * 20
        if y > control_bottom - 4:
            break
        color = (90, 220, 255) if index == 0 else (230, 230, 230)
        cv2.putText(canvas, text_line, (control_left + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text_line, (control_left + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    color, 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    # FoundationPose/Cutie use the root logger for verbose per-frame tracing.
    # Keep those messages out of the real-time path while retaining this
    # runtime's named status logger at INFO.
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    logging.getLogger().setLevel(logging.WARNING)
    LOGGER.setLevel(logging.INFO)
    rig = load_rig(args.calibration.resolve())
    cube = load_cube(args.cube_config.resolve())
    locked = load_locked(args.locked_reference.resolve(), rig.calibration_id)
    if locked.name == cube.name:
        raise SystemExit("locked reference and cube object names must differ")
    for path in (args.mesh, args.calibration, args.cube_config, args.locked_reference):
        if not path.is_file():
            raise SystemExit(f"required file is missing: {path}")
    try:
        import pyrealsense2 as rs
        import torch
    except ImportError as exc:
        raise SystemExit(f"pose environment dependency is missing: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("FoundationPose++ requires an available CUDA GPU")

    detector = CubeTagDetector(cube, args.tag_reprojection_p95_px)
    debug_parent = MOTION_ROOT / "runtime"
    debug_parent.mkdir(parents=True, exist_ok=True)
    debug_runtime = tempfile.TemporaryDirectory(
        prefix="foundationpose-debug-", dir=debug_parent
    )
    estimator, to_origin, bbox, draw_posed_3d_box, draw_xyz_axis = create_estimator(
        args.mesh.resolve(), args.mesh_scale, Path(debug_runtime.name),
        cube_symmetry=False,
    )
    publisher = DatagramPublisher(args.socket_path.resolve(), rig.calibration_id,
                                  (locked.name, cube.name))
    pipelines = []
    local = None
    bridge_mesh_cube = None
    mesh_pose = None
    cube_pose_camera0 = None
    previous_cube_world = None
    previous_cube_ns = None
    invalid_frames = 0
    disagreement_frames = 0
    frame_index = 0
    camera_restarts = 0
    last_bgr1 = None
    last_camera1_frame_ns = None
    last_tag_pose = None
    tag_roi_full = None
    visible_ids: list[int] = []
    tag_p95 = float("inf")
    last_reason = "waiting_for_cube_tags"
    cube_covariance = np.diag([0.006**2] * 3 + [math.radians(4.0)**2] * 3).reshape(-1).tolist()
    ui: TrackingWindow | None = None
    try:
        try:
            (pipe0, align0, depth_scale, pipe1, K0, D0, K1, D1,
             rgb_width, rgb_height, capture_fps,
             intrinsics_source) = start_cameras(rs, rig, args)
        except RuntimeError as exc:
            if "busy" in str(exc).lower() or "errno=16" in str(exc).lower():
                raise SystemExit(
                    "camera_0 is already in use. Stop the previous "
                    "track_native/live_*pose/realsense-viewer process, then retry."
                ) from exc
            raise
        pipelines.extend((pipe0, pipe1))
        initial_top_height = round(rgb_height * args.display_width / rgb_width)
        initial_scene_height = (
            args.scene_height if args.scene_height > 0 else
            initial_top_height
        )
        initial_window_width = args.display_width * 2
        initial_window_height = initial_top_height + initial_scene_height
        window_name = "FoundationPose++ monocular cube tracking"
        scene_view = SceneViewState()
        scene_from_world = (
            np.linalg.inv(locked.tag_transform)
            if args.scene_frame == "locked-tag" else np.eye(4, dtype=np.float64)
        )
        scene_view.set_default_from_eye(
            (scene_from_world @ rig.t_rig_camera0)[:3, 3]
        )
        # Keep the reset view close to camera0's sight line, but offset it
        # slightly so the camera0 frustum itself remains visible.
        scene_view.default_yaw += math.radians(20.0)
        scene_view.reset()

        def ui_dimensions() -> tuple[int, int, int]:
            if ui is None:
                total_width, total_height = (initial_window_width,
                                             initial_window_height)
            else:
                total_width, total_height = ui.size()
            top_fraction = initial_top_height / initial_window_height
            top_height = int(round(total_height * top_fraction))
            if total_height >= 240:
                top_height = int(np.clip(top_height, 120, total_height - 120))
            else:
                top_height = max(1, total_height // 2)
            current_scene_height = max(1, total_height - top_height)
            scene_view.scene_top = top_height
            scene_view.scene_width = total_width
            scene_view.scene_height = current_scene_height
            return total_width, top_height, current_scene_height

        if not args.no_display:
            ui = TrackingWindow(
                window_name, initial_window_width, initial_window_height,
                scene_view,
            )
        LOGGER.info("camera_0=%s is the only pose source; camera_1=%s is projection-only",
                     rig.serial0, rig.serial1)
        LOGGER.info(
            "capture RGB=%dx%d@%d, depth=%dx%d@%d; intrinsics=%s",
            rgb_width, rgb_height, capture_fps,
            args.depth_width, args.depth_height, capture_fps,
            intrinsics_source,
        )
        LOGGER.info(
            "active camera_0 K=[fx=%.3f fy=%.3f cx=%.3f cy=%.3f], "
            "camera_1 K=[fx=%.3f fy=%.3f cx=%.3f cy=%.3f]",
            K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2],
            K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2],
        )
        LOGGER.info("DR-NAS IPC destination: %s", args.socket_path.resolve())
        LOGGER.info("camera_0 inference scale=%.3f (%dx%d input)",
                     args.inference_scale, round(rgb_width * args.inference_scale),
                     round(rgb_height * args.inference_scale))
        LOGGER.info(
            "initial UI layout: top camera panes=%dx%d each, bottom 3-D scene=%dx%d",
            args.display_width,
            initial_top_height,
            initial_window_width, initial_scene_height,
        )
        LOGGER.info(
            "3-D display frame=%s%s (visualization only; tracking/IPC remain world)",
            args.scene_frame,
            (f", tag_{locked.tag_id} outward +Z is up"
             if args.scene_frame == "locked-tag" else ""),
        )
        LOGGER.info("waiting for cube tag IDs %s to initialize ROI", sorted(cube.object_points))
        while True:
            loop_started_ns = time.monotonic_ns()
            try:
                raw0 = newest_frames(pipe0, "camera_0", args.frame_timeout_ms)
            except CameraFrameTimeout as exc:
                publisher.lost(locked.name, "camera0_stream_timeout")
                publisher.lost(cube.name, "camera0_stream_timeout")
                publisher.heartbeat()
                camera_restarts += 1
                LOGGER.error(
                    "%s; restarting both RealSense streams (%d/%d)",
                    exc, camera_restarts, args.max_camera_restarts,
                )
                if camera_restarts > args.max_camera_restarts:
                    raise SystemExit(
                        f"camera_0 stopped delivering frames after "
                        f"{args.max_camera_restarts} stream restarts. Both configured "
                        "D435i cameras share USB hub 4-8; move one camera to a different "
                        "USB root port or reduce the calibrated stream profile."
                    ) from exc
                for pipeline in reversed(pipelines):
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                pipelines.clear()
                time.sleep(0.5)
                try:
                    (pipe0, align0, depth_scale, pipe1, K0, D0, K1, D1,
                     rgb_width, rgb_height, capture_fps,
                     intrinsics_source) = start_cameras(rs, rig, args)
                except RuntimeError as restart_exc:
                    raise SystemExit(
                        f"RealSense stream restart failed: {restart_exc}"
                    ) from restart_exc
                pipelines.extend((pipe0, pipe1))
                mesh_pose = cube_pose_camera0 = None
                previous_cube_world = None
                previous_cube_ns = None
                invalid_frames = disagreement_frames = 0
                last_bgr1 = None
                last_camera1_frame_ns = None
                tag_roi_full = None
                last_reason = "camera_stream_restarted_waiting_for_tags"
                continue

            # Camera 1 is visualization-only. It must never block or terminate
            # the camera-0 pose-source loop.
            raw1 = poll_newest_frames(pipe1)
            frames0 = align0.process(raw0)
            color0, depth0 = frames0.get_color_frame(), frames0.get_depth_frame()
            if not color0 or not depth0:
                publisher.heartbeat()
                continue
            bgr0 = np.asanyarray(color0.get_data())
            color1 = None if raw1 is None else raw1.get_color_frame()
            if color1:
                last_bgr1 = np.asanyarray(color1.get_data()).copy()
                last_camera1_frame_ns = time.monotonic_ns()
            if last_bgr1 is None:
                bgr1 = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
            else:
                bgr1 = last_bgr1
            depth = np.asanyarray(depth0.get_data()).astype(np.float32) * depth_scale
            inference_size = (
                round(bgr0.shape[1] * args.inference_scale),
                round(bgr0.shape[0] * args.inference_scale),
            )
            bgr0_inference = cv2.resize(
                bgr0, inference_size, interpolation=cv2.INTER_AREA
            )
            rgb0_inference = cv2.cvtColor(bgr0_inference, cv2.COLOR_BGR2RGB)
            depth_inference = cv2.resize(
                depth, inference_size, interpolation=cv2.INTER_NEAREST
            )
            K0_inference = K0.copy()
            K0_inference[0, :] *= inference_size[0] / bgr0.shape[1]
            K0_inference[1, :] *= inference_size[1] / bgr0.shape[0]
            timestamp = frame_timestamp(rs, color0)
            frame_number = int(color0.get_frame_number())
            frame_index += 1

            should_detect = mesh_pose is None or frame_index % args.tag_check_period == 0
            if should_detect:
                last_tag_pose, visible_ids, tag_p95 = detector.detect(bgr0, K0, D0)
                tag_roi_full = (
                    None if last_tag_pose is None else
                    initialization_mask(bgr0.shape[:2], cube, last_tag_pose,
                                        K0, D0)
                )
                if mesh_pose is None:
                    if not detector.raw_ids:
                        last_reason = "waiting_for_cube_tags"
                    elif not visible_ids:
                        last_reason = "decoded_non_cube_tag"
                    elif last_tag_pose is None:
                        last_reason = "cube_tag_pose_rejected"
                    else:
                        last_reason = "cube_tag_pose_ready"

            need_register = mesh_pose is None
            if mesh_pose is not None and last_tag_pose is not None and should_detect:
                translation_jump, rotation_jump = pose_jump_metrics(cube_pose_camera0, last_tag_pose)
                disagrees = (translation_jump * 1000.0 > args.reinit_translation_mm or
                             rotation_jump > args.reinit_rotation_deg)
                disagreement_frames = disagreement_frames + 1 if disagrees else 0
                if disagrees:
                    LOGGER.warning(
                        "tag/track disagreement %.1f mm, %.1f deg (%d/%d)",
                        translation_jump * 1000.0, rotation_jump,
                        disagreement_frames, args.reinit_confirmations,
                    )
                if disagreement_frames >= args.reinit_confirmations:
                    need_register = True
                    last_reason = "tag_disagreement_reinitializing"
            if invalid_frames >= args.max_invalid_frames:
                need_register = True

            if need_register and last_tag_pose is not None:
                mask = initialization_mask(
                    depth_inference.shape, cube, last_tag_pose,
                    K0_inference, D0,
                )
                valid_depth = 0 if mask is None else int(np.count_nonzero(
                    (mask > 0) & (depth_inference > 1e-4)
                ))
                if mask is None or valid_depth < 30:
                    last_reason = f"tag_roi_insufficient_depth:{valid_depth}"
                elif bridge_mesh_cube is not None:
                    try:
                        desired_mesh = last_tag_pose @ np.linalg.inv(bridge_mesh_cube)
                        seed_estimator_public_pose(estimator, desired_mesh)
                        mesh_pose = desired_mesh
                        cube_pose_camera0 = last_tag_pose.copy()
                        local = reset_local(
                            local, estimator, rgb0_inference, mask,
                            timestamp[1] if timestamp else time.monotonic_ns(),
                            not args.no_2d_tracker, not args.no_kalman_filter,
                        )
                        invalid_frames = disagreement_frames = 0
                        last_reason = "fast_reseed_from_cube_tags"
                        LOGGER.warning(
                            "FoundationPose/Cutie/Kalman fast-reseeded from tags %s "
                            "(p95=%.2f px, no global pose search)",
                            visible_ids, tag_p95,
                        )
                    except Exception as exc:  # hardware/CUDA dependent
                        mesh_pose = cube_pose_camera0 = None
                        last_reason = f"fast_reseed_failed:{type(exc).__name__}"
                        LOGGER.warning("FoundationPose fast reseed failed: %s", exc)
                else:
                    try:
                        last_reason = "registering_foundationpose"
                        LOGGER.info(
                            "cube tags %s accepted (p95=%.2f px, ROI depth pixels=%d); "
                            "starting FoundationPose registration",
                            visible_ids, tag_p95, valid_depth,
                        )
                        if not args.no_display:
                            ui_width, ui_top_height, ui_scene_height = ui_dimensions()
                            preview = draw_roi(bgr0, tag_roi_full)
                            if detector.raw_ids:
                                preview_ids = np.asarray(
                                    detector.raw_ids, dtype=np.int32
                                ).reshape(-1, 1)
                                cv2.aruco.drawDetectedMarkers(
                                    preview, detector.raw_corners, preview_ids,
                                    borderColor=(255, 255, 0),
                                )
                            cv2.putText(
                                preview,
                                "ROI READY - FoundationPose REGISTERING...",
                                (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                                (0, 0, 255), 2, cv2.LINE_AA,
                            )
                            waiting_right = bgr1.copy()
                            cv2.putText(
                                waiting_right, "camera1 projection waits for pose",
                                (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 255), 2, cv2.LINE_AA,
                            )
                            ui.show(
                                combined_view(
                                    preview, waiting_right,
                                    ui_width, ui_top_height,
                                    render_rig_scene(
                                        ui_width, ui_scene_height, rig, locked,
                                        cube,
                                        rig.t_rig_camera0 @ last_tag_pose,
                                        "registering_foundationpose",
                                        K0, K1, rgb_width, rgb_height,
                                        args.scene_frame, scene_view,
                                    ),
                                )
                            )
                        registration_started = time.monotonic()
                        registered = np.asarray(estimator.register(
                            K=K0_inference.astype(np.float32),
                            rgb=rgb0_inference, depth=depth_inference,
                            ob_mask=mask, iteration=args.est_refine_iter,
                        ), dtype=np.float64)
                        if bridge_mesh_cube is None:
                            bridge_mesh_cube = np.linalg.inv(registered) @ last_tag_pose
                        else:
                            desired_mesh = last_tag_pose @ np.linalg.inv(bridge_mesh_cube)
                            registered = canonicalize_cube_pose(
                                registered, desired_mesh, estimator.model_center
                            )
                        mesh_pose = registered
                        cube_pose_camera0 = mesh_pose @ bridge_mesh_cube
                        local = reset_local(
                            local, estimator, rgb0_inference, mask,
                            timestamp[1] if timestamp else time.monotonic_ns(),
                            not args.no_2d_tracker, not args.no_kalman_filter,
                        )
                        invalid_frames = disagreement_frames = 0
                        last_reason = "registered_from_cube_tags"
                        LOGGER.warning(
                            "FoundationPose registered from tags %s (p95=%.2f px) "
                            "in %.2f s",
                            visible_ids, tag_p95,
                            time.monotonic() - registration_started,
                        )
                    except Exception as exc:  # hardware/CUDA dependent
                        mesh_pose = cube_pose_camera0 = None
                        last_reason = f"registration_failed:{type(exc).__name__}"
                        LOGGER.warning("FoundationPose registration failed: %s", exc)
            elif mesh_pose is not None and not need_register:
                try:
                    local.set_timestamp(timestamp[1] if timestamp else time.monotonic_ns(), capture_fps)
                    candidate, _, track_error = local.track(
                        estimator, rgb0_inference, depth_inference,
                        K0_inference.astype(np.float32), args.track_refine_iter
                    )
                    if track_error:
                        raise RuntimeError(track_error)
                    candidate = canonicalize_cube_pose(
                        np.asarray(candidate, dtype=np.float64), mesh_pose, estimator.model_center
                    )
                    candidate_cube = candidate @ bridge_mesh_cube
                    plausible, reason = pose_is_plausible(
                        candidate_cube, cube, depth, K0, D0,
                        args.min_depth_m, args.max_depth_m,
                    )
                    if not plausible:
                        raise RuntimeError(reason)
                    mesh_pose, cube_pose_camera0 = candidate, candidate_cube
                    invalid_frames = 0
                    last_reason = "foundationpose_camera0"
                except Exception as exc:  # hardware/CUDA dependent
                    invalid_frames += 1
                    last_reason = f"tracking_invalid:{exc}"
                    LOGGER.warning("camera_0 tracking invalid (%d/%d): %s",
                                    invalid_frames, args.max_invalid_frames, exc)
                    if invalid_frames >= args.max_invalid_frames:
                        mesh_pose = cube_pose_camera0 = None

            cube_world = None if cube_pose_camera0 is None else rig.t_rig_camera0 @ cube_pose_camera0
            cube_camera1 = None if cube_world is None else np.linalg.inv(rig.t_rig_camera1) @ cube_world
            mesh_world = None if mesh_pose is None else rig.t_rig_camera0 @ mesh_pose
            mesh_camera1 = None if mesh_world is None else np.linalg.inv(rig.t_rig_camera1) @ mesh_world
            if timestamp is not None:
                publisher.measured(
                    locked.name, locked.transform, locked.covariance, [0.0] * 6,
                    timestamp, frame_number, [], "locked_reference_camera0_clock",
                )
                if cube_world is not None:
                    twist = twist_between(previous_cube_world, cube_world,
                                          previous_cube_ns, timestamp[1])
                    publisher.measured(
                        cube.name, cube_world, cube_covariance, twist, timestamp,
                        frame_number, visible_ids if should_detect else [],
                        "foundationpose_camera0",
                    )
                    previous_cube_world = cube_world.copy()
                    previous_cube_ns = timestamp[1]
                else:
                    publisher.lost(cube.name, last_reason)
            else:
                publisher.lost(cube.name, "camera0_timestamp_not_authoritative")
            publisher.heartbeat()

            if not args.no_display:
                left_base = (draw_roi(bgr0, tag_roi_full)
                             if mesh_pose is None else bgr0)
                left = draw_foundationpose_overlay(
                    left_base, mesh_pose, cube_pose_camera0,
                    to_origin, bbox, K0,
                    draw_posed_3d_box, draw_xyz_axis,
                    "FP++ camera0", (0, 255, 0),
                )
                if detector.raw_ids:
                    marker_ids = np.asarray(detector.raw_ids, dtype=np.int32).reshape(-1, 1)
                    cv2.aruco.drawDetectedMarkers(
                        left, detector.raw_corners, marker_ids,
                        borderColor=(255, 255, 0),
                    )
                right = draw_foundationpose_overlay(
                    bgr1, mesh_camera1, cube_camera1,
                    to_origin, bbox, K1,
                    draw_posed_3d_box, draw_xyz_axis,
                    "camera1 projected", (255, 255, 0),
                )
                cv2.putText(left, f"camera0 POSE SOURCE | {last_reason}", (18, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                timestamp_text = "IPC timestamp: authoritative" if timestamp else "IPC timestamp: NOT AUTHORITATIVE"
                cv2.putText(left, timestamp_text, (18, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255) if timestamp else (0, 0, 255), 2, cv2.LINE_AA)
                now_ns = time.monotonic_ns()
                age_ms = float("inf") if timestamp is None else (now_ns - timestamp[1]) * 1e-6
                timing_text = f"age={age_ms:.1f} ms  loop={(now_ns-loop_started_ns)*1e-6:.1f} ms"
                cv2.putText(left, timing_text, (18, 95), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255) if age_ms <= 67.0 else (0, 0, 255),
                            2, cv2.LINE_AA)
                p95_text = "inf" if not np.isfinite(tag_p95) else f"{tag_p95:.1f}px"
                tag_text = (f"raw tags={detector.raw_ids}  cube tags={visible_ids}  "
                            f"expected={sorted(cube.object_points)}  p95={p95_text}")
                cv2.putText(left, tag_text, (18, 125), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 0) if detector.raw_ids else (0, 0, 255),
                            2, cv2.LINE_AA)
                cv2.putText(right, "camera1 VISUALIZATION ONLY", (18, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(right, f"locked palm: {locked.name}", (18, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                ipc_text = "DR-NAS IPC: connected" if publisher.receiver_present else "DR-NAS IPC: waiting for consumer"
                cv2.putText(right, ipc_text, (18, 95), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2, cv2.LINE_AA)
                camera1_age_ms = (float("inf") if last_camera1_frame_ns is None else
                                  (time.monotonic_ns() - last_camera1_frame_ns) * 1e-6)
                camera1_text = ("camera1 stream: OK" if camera1_age_ms <= 1000.0 else
                                "camera1 stream: STALLED (tracking unaffected)")
                cv2.putText(right, camera1_text, (18, 125), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255) if camera1_age_ms <= 1000.0 else (0, 0, 255),
                            2, cv2.LINE_AA)
                ui_width, ui_top_height, ui_scene_height = ui_dimensions()
                scene = render_rig_scene(
                    ui_width, ui_scene_height,
                    rig, locked, cube, cube_world, last_reason,
                    K0, K1, rgb_width, rgb_height,
                    args.scene_frame,
                    scene_view,
                )
                ui.show(
                    combined_view(left, right, ui_width, ui_top_height, scene)
                )
                key = ui.poll_key()
                if ui.closed or key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    mesh_pose = cube_pose_camera0 = None
                    invalid_frames = disagreement_frames = 0
                    tag_roi_full = None
                    last_reason = "manual_reinitialization"
                if key in (ord("v"), ord("V"), ord("0")):
                    scene_view.reset()
    finally:
        try:
            publisher.lost(cube.name, "producer_stopped")
        except Exception as exc:
            LOGGER.warning("could not publish final LOST event: %s", exc)
        try:
            publisher.close()
        except Exception as exc:
            LOGGER.warning("could not close IPC publisher: %s", exc)
        for pipeline in reversed(pipelines):
            try:
                pipeline.stop()
            except Exception:
                pass
        if ui is not None:
            ui.close()
        mesh_export_tmp = getattr(estimator, "_mesh_export_tmp", None)
        if mesh_export_tmp is not None:
            mesh_export_tmp.cleanup()
        debug_runtime.cleanup()
    return 0


if __name__ == "__main__":
    def _request_shutdown(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested; RealSense pipelines released")
        raise SystemExit(130)
