"""Rig-frame fusion of independent FoundationPose camera measurements.

FoundationPose returns ``T_camera_object``.  This module deliberately keeps
the camera geometry separate from filtering:

* ``T_rig_camera`` converts each camera result into the calibrated rig frame;
* one shared :class:`KalmanFilter6D` is updated once per valid camera result;
* the innovation gate rejects an inconsistent camera instead of averaging a
  bad pose into the good one.

The module has no ROS or RealSense dependency, so it can be used by a live
script, a ROS node, or an offline recording evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .kalman_filter_6d import KalmanFilter6D


def _as_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise ValueError(f"{name} rotation is not a proper orthonormal matrix")
    return transform


def transform_camera_pose_to_rig(
    t_rig_camera: np.ndarray, t_camera_object: np.ndarray
) -> np.ndarray:
    """Convert FoundationPose's camera-frame pose to the rig frame."""
    return _as_transform(t_rig_camera, "t_rig_camera") @ _as_transform(
        t_camera_object, "t_camera_object"
    )


def pose_to_6d(t_rig_object: np.ndarray) -> np.ndarray:
    transform = _as_transform(t_rig_object, "t_rig_object")
    return np.r_[
        transform[:3, 3],
        Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"),
    ]


def sixd_to_pose(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("pose state must be a finite six-element vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def rotate_pose_covariance(
    t_rig_camera: np.ndarray, covariance_camera: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Rotate a camera-frame [translation, small-angle] covariance to rig.

    This is the first-order covariance transform for a fixed camera extrinsic.
    It is exact for the translational and rotational blocks when the same
    world-frame small-angle convention is used by both sides.
    """
    if covariance_camera is None:
        return None
    covariance = np.asarray(covariance_camera, dtype=np.float64)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite 6x6 matrix")
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(np.diag(covariance) <= 0.0):
        raise ValueError("covariance diagonal must be positive")
    rotation = _as_transform(t_rig_camera, "t_rig_camera")[:3, :3]
    jacobian = np.zeros((6, 6), dtype=np.float64)
    jacobian[:3, :3] = rotation
    jacobian[3:, 3:] = rotation
    output = jacobian @ covariance @ jacobian.T
    output += np.eye(6) * 1e-12
    return output


@dataclass(frozen=True)
class PoseMeasurement:
    """One camera measurement in the camera coordinate convention."""

    source: str
    acquisition_stamp_ns: int
    t_camera_object: np.ndarray
    t_rig_camera: np.ndarray
    covariance: Optional[np.ndarray] = None

    def in_rig(self) -> "RigPoseMeasurement":
        t_rig_camera = _as_transform(self.t_rig_camera, "t_rig_camera")
        return RigPoseMeasurement(
            source=self.source,
            acquisition_stamp_ns=int(self.acquisition_stamp_ns),
            t_rig_object=transform_camera_pose_to_rig(
                t_rig_camera, self.t_camera_object
            ),
            covariance=rotate_pose_covariance(t_rig_camera, self.covariance),
        )


@dataclass(frozen=True)
class RigPoseMeasurement:
    source: str
    acquisition_stamp_ns: int
    t_rig_object: np.ndarray
    covariance: Optional[np.ndarray] = None


@dataclass
class FusionResult:
    t_rig_object: np.ndarray
    state: np.ndarray
    accepted_sources: list[str]
    rejected_sources: list[str]
    innovation_distances: dict[str, float]
    timestamp_ns: int


@dataclass(frozen=True)
class FusionSnapshot:
    """Copy of the filter state used to roll back an invalid update."""

    mean: Optional[np.ndarray]
    covariance: Optional[np.ndarray]
    timestamp_ns: Optional[int]


class StereoPoseFusion:
    """One shared 6D filter for two (or more) independent cameras."""

    def __init__(
        self,
        measurement_noise_scale: float = 0.05,
        innovation_gate: float = 16.812,
        max_pair_skew_ns: int = 20_000_000,
        default_translation_std_m: float = 0.003,
        default_rotation_std_deg: float = 1.0,
    ) -> None:
        if not np.isfinite(innovation_gate) or innovation_gate <= 0:
            raise ValueError("innovation_gate must be finite and positive")
        if max_pair_skew_ns < 0:
            raise ValueError("max_pair_skew_ns must be non-negative")
        if (
            not np.isfinite(default_translation_std_m)
            or default_translation_std_m <= 0.0
            or not np.isfinite(default_rotation_std_deg)
            or default_rotation_std_deg <= 0.0
        ):
            raise ValueError("default measurement standard deviations must be positive")
        self.filter = KalmanFilter6D(measurement_noise_scale)
        self.innovation_gate = float(innovation_gate)
        self.max_pair_skew_ns = int(max_pair_skew_ns)
        default_rotation_std = np.deg2rad(float(default_rotation_std_deg))
        self.default_measurement_covariance = np.diag(
            [
                float(default_translation_std_m) ** 2,
                float(default_translation_std_m) ** 2,
                float(default_translation_std_m) ** 2,
                default_rotation_std**2,
                default_rotation_std**2,
                default_rotation_std**2,
            ]
        )
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.timestamp_ns: Optional[int] = None

    @property
    def initialized(self) -> bool:
        return self.mean is not None and self.covariance is not None

    def reset(self) -> None:
        self.mean = None
        self.covariance = None
        self.timestamp_ns = None

    def snapshot(self) -> FusionSnapshot:
        """Return a safe copy of the current state before prediction/update."""
        return FusionSnapshot(
            mean=None if self.mean is None else self.mean.copy(),
            covariance=None if self.covariance is None else self.covariance.copy(),
            timestamp_ns=self.timestamp_ns,
        )

    def restore(self, snapshot: FusionSnapshot) -> None:
        """Restore a state captured by :meth:`snapshot`."""
        self.mean = None if snapshot.mean is None else snapshot.mean.copy()
        self.covariance = (
            None if snapshot.covariance is None else snapshot.covariance.copy()
        )
        self.timestamp_ns = snapshot.timestamp_ns

    def hold_current(self) -> None:
        """Stop constant-velocity extrapolation after rejected measurements."""
        if self.mean is not None:
            # A rejected observation is not evidence of motion. Keeping the
            # old velocity here would make every subsequent prediction drift
            # farther from the last trusted pose while the cameras recover.
            self.mean[6:] = 0.0

    def _predict_to(self, timestamp_ns: int) -> None:
        if not self.initialized:
            return
        assert self.timestamp_ns is not None
        if timestamp_ns < self.timestamp_ns:
            raise ValueError("measurement timestamp moved backwards")
        dt = (timestamp_ns - self.timestamp_ns) * 1e-9
        assert self.mean is not None and self.covariance is not None
        self.mean, self.covariance = self.filter.predict(
            self.mean, self.covariance, dt=dt
        )
        self.timestamp_ns = timestamp_ns

    def predict_to(self, timestamp_ns: int) -> Optional[np.ndarray]:
        """Predict the shared state to ``timestamp_ns`` and return its pose.

        The live dual-camera adapter uses this prediction as the prior for
        both camera estimators before collecting their measurements.
        """
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if not self.initialized:
            return None
        self._predict_to(timestamp_ns)
        assert self.mean is not None
        return sixd_to_pose(self.mean[:6])

    def current_pose(self) -> Optional[np.ndarray]:
        """Return the current filtered rig pose, or ``None`` before init."""
        if not self.initialized:
            return None
        assert self.mean is not None
        return sixd_to_pose(self.mean[:6])

    def result_from_state(
        self,
        timestamp_ns: Optional[int],
        rejected_sources: Sequence[str] = (),
        innovation_distances: Optional[dict[str, float]] = None,
    ) -> FusionResult:
        """Build a result without changing the filter state.

        The live adapter uses this after an all-camera gate rejection. The
        previous filtered pose is held instead of publishing a prediction
        driven only by an outdated velocity estimate.
        """
        if not self.initialized or self.mean is None or self.covariance is None:
            raise ValueError("cannot build a result before fusion is initialized")
        state_timestamp = self.timestamp_ns if timestamp_ns is None else int(timestamp_ns)
        if state_timestamp is None:
            raise ValueError("fusion state has no timestamp")
        return FusionResult(
            t_rig_object=sixd_to_pose(self.mean[:6]),
            state=self.mean.copy(),
            accepted_sources=[],
            rejected_sources=list(rejected_sources),
            innovation_distances=dict(innovation_distances or {}),
            timestamp_ns=state_timestamp,
        )

    def fuse(self, measurements: Iterable[PoseMeasurement]) -> FusionResult:
        rig_measurements = [measurement.in_rig() for measurement in measurements]
        if not rig_measurements:
            raise ValueError("at least one pose measurement is required")
        if any(item.acquisition_stamp_ns <= 0 for item in rig_measurements):
            raise ValueError("measurement timestamps must be positive")
        rig_measurements.sort(key=lambda item: item.acquisition_stamp_ns)
        first_stamp = rig_measurements[0].acquisition_stamp_ns
        last_stamp = rig_measurements[-1].acquisition_stamp_ns
        if last_stamp - first_stamp > self.max_pair_skew_ns:
            raise ValueError(
                "camera measurements exceed the configured stereo time skew"
            )

        accepted: list[str] = []
        rejected: list[str] = []
        distances: dict[str, float] = {}
        for measurement in rig_measurements:
            stamp = measurement.acquisition_stamp_ns
            if self.timestamp_ns is not None and stamp < self.timestamp_ns:
                # A late packet from a previous pair cannot be applied to a
                # state that has already advanced.  The caller can retain it
                # for diagnostics, but dropping it here avoids a time-travel
                # update and matches the ROS acquisition reorder policy.
                rejected.append(measurement.source)
                distances[measurement.source] = float("inf")
                continue
            covariance = (
                measurement.covariance
                if measurement.covariance is not None
                else self.default_measurement_covariance
            )
            if not self.initialized:
                self.mean, self.covariance = self.filter.initiate(
                    pose_to_6d(measurement.t_rig_object), covariance
                )
                self.timestamp_ns = stamp
                accepted.append(measurement.source)
                distances[measurement.source] = 0.0
                continue

            self._predict_to(stamp)
            assert self.mean is not None and self.covariance is not None
            distance = self.filter.innovation_distance(
                self.mean,
                self.covariance,
                pose_to_6d(measurement.t_rig_object),
                covariance,
            )
            distances[measurement.source] = distance
            old_mean, old_covariance = self.mean, self.covariance
            self.mean, self.covariance = self.filter.update(
                self.mean,
                self.covariance,
                pose_to_6d(measurement.t_rig_object),
                measurement_covariance=covariance,
                gate_threshold=self.innovation_gate,
            )
            if np.array_equal(self.mean, old_mean) and np.array_equal(
                self.covariance, old_covariance
            ):
                rejected.append(measurement.source)
            else:
                accepted.append(measurement.source)

        assert self.mean is not None and self.covariance is not None
        assert self.timestamp_ns is not None
        return FusionResult(
            t_rig_object=sixd_to_pose(self.mean[:6]),
            state=self.mean.copy(),
            accepted_sources=accepted,
            rejected_sources=rejected,
            innovation_distances=distances,
            timestamp_ns=self.timestamp_ns,
        )
