#!/usr/bin/env python3
"""Run FoundationPose++ on live frames from an Intel RealSense D435.

The first frame is initialized from a user-selected rectangle.  Subsequent
frames use FoundationPose's previous-pose tracker; enabling ``--activate-2d-
tracker`` adds the documented FoundationPose++ 2D correction stage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=Path("test/mesh/black_cube.STL"))
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=0.001,
        help="Scale applied to mesh coordinates before inference (STL files here are mm).",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--depth-width",
        type=int,
        help="Depth stream width (defaults to --width). Use 1280 with 1920x1080 color.",
    )
    parser.add_argument(
        "--depth-height",
        type=int,
        help="Depth stream height (defaults to --height). Use 720 with 1920x1080 color.",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="254322070766",
        help="RealSense serial number (default: camera_0 254322070766).",
    )
    parser.add_argument("--est-refine-iter", type=int, default=10)
    parser.add_argument("--track-refine-iter", type=int, default=3)
    parser.add_argument("--activate-2d-tracker", action="store_true", help="Use Cutie to track the object bbox between frames.")
    parser.add_argument(
        "--activate-kalman-filter",
        action="store_true",
        help="Smooth the 6D pose and fuse the optional 2D tracker measurement.",
    )
    parser.add_argument("--kf-measurement-noise-scale", type=float, default=0.05)
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        default=[0, 159, 237],
        metavar=("R", "G", "B"),
        help="Fallback RGB texture for colorless meshes.",
    )
    parser.add_argument("--output", type=Path, default=Path("test/live_pose.npy"))
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Initial object rectangle in pixels; enables headless operation.",
    )
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window (requires --roi).")
    return parser.parse_args()


def make_mask(frame_shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=bool)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + width), min(h, y + height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def select_and_register(estimator, K, rgb, depth, refine_iter, roi=None):
    """Select an ROI (or use the supplied rectangle) and register."""
    if roi is None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x, y, width, height = cv2.selectROI("FoundationPose++", bgr, showCrosshair=True)
    else:
        x, y, width, height = roi
    if width < 2 or height < 2:
        return None, None
    mask = make_mask(rgb.shape[:2], (int(x), int(y), int(width), int(height)))
    if np.count_nonzero(mask & (depth > 1e-4)) < 20:
        logging.warning("The selected ROI has too few valid depth pixels; select it again.")
        return None, None
    pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=refine_iter)
    return pose, mask


def adjust_pose_to_image_point(pose, K, x, y):
    """Move only the pose translation so its origin projects to (x, y)."""
    batched = pose.ndim == 3
    pose = pose[0].clone() if batched else pose.clone()
    z = pose[2, 3]
    pose[0, 3] = (float(x) - float(K[0, 2])) * z / float(K[0, 0])
    pose[1, 3] = (float(y) - float(K[1, 2])) * z / float(K[1, 1])
    return pose.unsqueeze(0) if batched else pose


def pose_to_6d(pose):
    from scipy.spatial.transform import Rotation

    pose_np = pose.detach().cpu().numpy() if hasattr(pose, "detach") else np.asarray(pose)
    if pose_np.ndim == 3:
        pose_np = pose_np[0]
    return np.r_[pose_np[:3, 3], Rotation.from_matrix(pose_np[:3, :3]).as_euler("xyz")]


def sixd_to_pose(values):
    from scipy.spatial.transform import Rotation

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix().astype(np.float32)
    pose[:3, 3] = np.asarray(values[:3], dtype=np.float32)
    return pose


def main() -> int:
    args = parse_args()
    if args.no_display and args.roi is None:
        raise SystemExit("--no-display requires --roi X Y W H so the first pose can be initialized.")
    repo_dir = Path(__file__).resolve().parents[1]
    foundationpose_dir = repo_dir / "FoundationPose"
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(foundationpose_dir))

    try:
        import pyrealsense2 as rs
        import torch
        import trimesh
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Install it in the 'pose' conda environment.") from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "FoundationPose requires a CUDA GPU, but torch.cuda.is_available() is false. "
            "Install/enable an NVIDIA driver before starting this script."
        )

    try:
        # These imports are intentionally delayed so a missing FoundationPose
        # dependency produces a useful error before the camera is opened.
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        from Utils import draw_posed_3d_box, draw_xyz_axis, dr, trimesh_add_pure_colored_texture
    except ImportError as exc:
        raise SystemExit(
            f"FoundationPose dependencies are incomplete: {exc}. "
            "Install PyTorch3D and nvdiffrast in the 'pose' environment."
        ) from exc

    tracker_2d = None
    kalman_filter = None
    if args.activate_2d_tracker:
        try:
            sys.path.insert(0, str(repo_dir / "src"))
            from VOT import Cutie

            tracker_2d = Cutie()
        except ImportError as exc:
            raise SystemExit(f"The 2D tracker is unavailable: {exc}. Install the Cutie dependencies in the pose environment.") from exc
    if args.activate_kalman_filter:
        try:
            sys.path.insert(0, str(repo_dir / "src"))
            from utils.kalman_filter_6d import LegacyKalmanFilter6D

            # Keep the native single-camera behavior equivalent to the JHU
            # reference. The time-aware filter is reserved for rig fusion.
            kalman_filter = LegacyKalmanFilter6D(args.kf_measurement_noise_scale)
        except ImportError as exc:
            raise SystemExit(f"The Kalman filter is unavailable: {exc}") from exc

    mesh_path = args.mesh if args.mesh.is_absolute() else repo_dir / args.mesh
    if not mesh_path.exists():
        raise SystemExit(f"Mesh not found: {mesh_path}")
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.apply_scale(args.mesh_scale)
    try:
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.asarray(args.color, dtype=np.uint8), resolution=10)
    except Exception as exc:
        logging.warning("Could not add a fallback mesh texture (%s); continuing.", exc)

    # RealSense intrinsics and FoundationPose tensors use float32.  STL data
    # commonly loads as float64, which otherwise causes a dtype mismatch in
    # the crop projection code during registration.
    # Trimesh's public vertex setter promotes back to float64, so update its
    # tracked data buffer directly and clear cached derived geometry.
    mesh._data['vertices'] = np.asarray(mesh.vertices, dtype=np.float32)
    mesh._cache.clear()

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
    )
    # FoundationPose centers a private trimesh copy during initialization;
    # trimesh promotes that assignment to float64. Normalize the copy used by
    # the refiner as well, without changing the upstream implementation.
    if estimator.mesh is not None:
        estimator.mesh._data['vertices'] = np.asarray(estimator.mesh.vertices, dtype=np.float32)
        estimator.mesh._cache.clear()
    # compute_mesh_diameter returns a NumPy scalar for point arrays.  In the
    # upstream crop helper, a NumPy float64 radius makes torch infer double
    # offsets and promotes otherwise-float32 pose tensors.  A Python float
    # follows torch's configured float32 default instead.
    estimator.diameter = float(estimator.diameter)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    depth_width = args.depth_width if args.depth_width is not None else args.width
    depth_height = args.depth_height if args.depth_height is not None else args.height
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, args.fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not start the RealSense pipeline: {exc}. "
            "Check that the D435 is connected and accessible to this user."
        ) from exc
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    K = np.array(
        [[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    device = profile.get_device()
    logging.info(
        "Using RealSense %s (serial=%s, firmware=%s)",
        device.get_info(rs.camera_info.name),
        device.get_info(rs.camera_info.serial_number),
        device.get_info(rs.camera_info.firmware_version),
    )
    logging.info(
        "Streams: color=%dx%d, depth=%dx%d, fps=%d (aligned depth follows color)",
        args.width,
        args.height,
        depth_width,
        depth_height,
        args.fps,
    )
    logging.info("D435 depth scale: %g m; K=\n%s", depth_scale, K)

    poses = []
    pose = None
    initialized = False
    init_mask = None
    bbox_2d = None
    kf_mean = kf_covariance = None
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

            key = -1
            if not args.no_display:
                preview = color_bgr.copy()
                if pose is not None:
                    # Keep the latest estimate visible while the next frame is
                    # being processed.  FoundationPose returns the pose in
                    # the centered-object frame, so undo the mesh centering
                    # transform before projecting the model box and axes.
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis_rgb = draw_posed_3d_box(K, img=cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), ob_in_cam=center_pose, bbox=bbox)
                    vis_rgb = draw_xyz_axis(vis_rgb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                    preview = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                    xyz = center_pose[:3, 3]
                    cv2.putText(preview, f"Pose  t=[{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}] m", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    if tracker_2d is not None:
                        cv2.putText(preview, "2D tracker active", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                        if bbox_2d is not None and bbox_2d[2] > 1 and bbox_2d[3] > 1:
                            x, y, width, height = map(int, bbox_2d)
                            cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 200, 0), 2)
                else:
                    cv2.putText(preview, "Press S to select object", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("FoundationPose++", preview)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("r")) or (not initialized and args.roi is not None):
                roi = args.roi if (not initialized and args.roi is not None) else None
                pose, init_mask = select_and_register(estimator, K, rgb, depth, args.est_refine_iter, roi=roi)
                initialized = pose is not None
                if initialized and tracker_2d is not None:
                    bbox_2d = tracker_2d.initialize(rgb, init_info={"mask": init_mask})
                if initialized and kalman_filter is not None:
                    kf_mean, kf_covariance = kalman_filter.initiate(
                        pose_to_6d(estimator.pose_last)
                    )
            elif initialized:
                if tracker_2d is not None:
                    bbox_2d = tracker_2d.track(rgb)
                    if bbox_2d is not None and bbox_2d[2] > 1 and bbox_2d[3] > 1:
                        x, y, width, height = bbox_2d
                        cx, cy = x + width / 2.0, y + height / 2.0
                        if kalman_filter is None:
                            estimator.pose_last = adjust_pose_to_image_point(
                                estimator.pose_last, K, cx, cy
                            )
                        else:
                            pose_last_single = (
                                estimator.pose_last[0]
                                if estimator.pose_last.ndim == 3
                                else estimator.pose_last
                            )
                            tz = float(pose_last_single[2, 3].detach().cpu())
                            if tz > 1e-5:
                                kf_mean, kf_covariance = kalman_filter.update(
                                    kf_mean, kf_covariance,
                                    pose_to_6d(estimator.pose_last)
                                )
                                measurement_xy = np.array(
                                    [
                                        (cx - K[0, 2]) * tz / K[0, 0],
                                        (cy - K[1, 2]) * tz / K[1, 1],
                                    ],
                                    dtype=np.float32,
                                )
                                kf_mean, kf_covariance = (
                                    kalman_filter.update_from_xy(
                                        kf_mean, kf_covariance, measurement_xy
                                    )
                                )
                                estimator.pose_last = torch.as_tensor(
                                    sixd_to_pose(kf_mean[:6]), device="cuda"
                                ).unsqueeze(0)
                pose = estimator.track_one(
                    rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter
                )
                if kalman_filter is not None:
                    kf_mean, kf_covariance = kalman_filter.predict(
                        kf_mean, kf_covariance
                    )

            if pose is None:
                continue
            poses.append(np.asarray(pose, dtype=np.float32).copy())
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        if poses:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output, np.stack(poses))
            np.savetxt(str(args.output) + ".K.txt", K)
            logging.info("Saved %d poses to %s", len(poses), args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
