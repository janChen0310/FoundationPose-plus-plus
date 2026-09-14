import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class LegacyKalmanFilter6D(object):
    """Compatibility implementation used by the original single-view FP++ path.

    The native JHU live adapter treats each processed image as one time step and
    uses this filter only to provide a local prior to ``track_one``.  The
    timestamp-aware :class:`KalmanFilter6D` below is intentionally different:
    it is used by the rig-frame multi-camera fusion code.  Keeping the two
    models separate prevents stereo work from changing single-camera behavior.
    """

    def __init__(self, measurement_noise_scale):
        ndim, dt = 6, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for index in range(ndim):
            self._motion_mat[index, ndim + index] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._update_mat_xy = np.zeros((2, 2 * ndim))
        self._update_mat_xy[0, 0] = 1
        self._update_mat_xy[1, 1] = 1
        self._std_weight_trans_xy = 1.0 / 40.0
        self._std_weight_trans = 1.0 / 10.0
        self._std_weight_rot = 1.0 / 20.0
        self._std_weight_vel_trans = 1.0 / 20.0
        self._std_weight_vel_rot = 1.0 / 40.0
        self.measurement_noise_scale = measurement_noise_scale

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        scale_xyz = max(np.linalg.norm(measurement[:3]), 1e-5)
        scale_rot = max(np.linalg.norm(measurement[3:]), 1e-5)
        std = [
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_rot * scale_xyz,
            self._std_weight_vel_rot * scale_xyz,
            self._std_weight_vel_rot * scale_xyz,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        scale_xyz = mean[2]
        scale_rot = mean[5]
        std_pos = [
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
        ]
        std_vel = [
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)
        ) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        scale_xyz = mean[2]
        scale_rot = mean[5]
        std = [
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
        ]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        ) + innovation_cov
        return projected_mean, projected_cov

    def project_for_xy(self, mean, covariance):
        scale_xy = max(np.linalg.norm(mean[:2]), 1e-5)
        innovation_cov = np.diag(
            np.square([self._std_weight_trans_xy * scale_xy] * 2)
        )
        projected_mean = np.dot(self._update_mat_xy, mean)
        projected_cov = np.linalg.multi_dot(
            (self._update_mat_xy, covariance, self._update_mat_xy.T)
        ) + innovation_cov
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def update_from_xy(self, mean, covariance, measurement_xy):
        projected_mean, projected_cov = self.project_for_xy(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat_xy.T).T,
            check_finite=False,
        ).T
        innovation = measurement_xy - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

class KalmanFilter6D(object):
    """
    6D Kalman filter for tracking 6DOF poses.
    
    State space (12-dimensional):
    [tx, ty, tz, rx, ry, rz, v_tx, v_ty, v_tz, v_rx, v_ry, v_rz]
    """
    
    def __init__(self, measurement_noise_scale):
        ndim, dt = 6, 1.0  # 6 pose dimensions
        
        # State transition matrix (12x12)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # Observation matrix (6x12)
        self._update_mat = np.eye(ndim, 2 * ndim)    # 6*12
        self._update_mat_xy = np.zeros((2, 2*ndim))  # 2*12
        # Diagonal
        self._update_mat_xy[0, 0] = 1
        self._update_mat_xy[1, 1] = 1

        # 新增xy传感器噪声参数
        self._std_weight_trans_xy = 1./40  # xy sensor has lower noise floor
        # Process noise parameters
        self._std_weight_trans = 1./10    # Translation uncertainty
        self._std_weight_rot = 1./20     # Rotation uncertainty
        self._std_weight_vel_trans = 1./20     # Translation speed uncertainty
        self._std_weight_vel_rot = 1./40    # Rotation speed uncertainty

        if not np.isfinite(measurement_noise_scale) or measurement_noise_scale <= 0:
            raise ValueError("measurement_noise_scale must be finite and positive")
        self.measurement_noise_scale = float(measurement_noise_scale)

    @staticmethod
    def _wrap_angle(value):
        """Return an angle in [-pi, pi), also accepting NumPy arrays."""
        return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _pose_innovation(cls, measurement, projected_mean):
        """Compute a 6D innovation without an Euler-angle 2*pi jump."""
        innovation = np.asarray(measurement, dtype=np.float64) - np.asarray(
            projected_mean, dtype=np.float64
        )
        innovation[3:6] = cls._wrap_angle(innovation[3:6])
        return innovation

    @staticmethod
    def _normalize_state_angles(state):
        state = np.asarray(state, dtype=np.float64).copy()
        state[3:6] = KalmanFilter6D._wrap_angle(state[3:6])
        return state

    @staticmethod
    def _validate_measurement_covariance(measurement_covariance):
        if measurement_covariance is None:
            return None
        covariance = np.asarray(measurement_covariance, dtype=np.float64)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("measurement_covariance must be a finite 6x6 matrix")
        covariance = 0.5 * (covariance + covariance.T)
        if np.any(np.diag(covariance) <= 0.0):
            raise ValueError("measurement_covariance diagonal must be positive")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement_covariance must be positive definite") from exc
        return covariance

    def initiate(self, measurement, measurement_covariance=None):
        """Initialize track with unassociated measurement (6DOF pose)"""
        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (6,) or not np.all(np.isfinite(measurement)):
            raise ValueError("measurement must be a finite six-element vector")
        measurement_covariance = self._validate_measurement_covariance(
            measurement_covariance
        )
        mean_pos = self._normalize_state_angles(measurement)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        # Initialize uncertainties
        scale_xyz = max(np.linalg.norm(measurement[:3]), 1e-5)
        scale_rot = max(np.linalg.norm(measurement[3:]), 1e-5)
        
        # 初始误差比较大
        std = [
            # Position uncertainties
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            
            # Velocity uncertainties
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_rot * scale_rot,
            1 * self._std_weight_vel_rot * scale_rot,
            1 * self._std_weight_vel_rot * scale_rot,
        ]
        covariance = np.diag(np.square(std))
        if measurement_covariance is not None:
            covariance[:6, :6] += measurement_covariance
        return mean, covariance

    def predict(self, mean, covariance, dt=1.0):
        """Run prediction for ``dt`` seconds (one legacy frame by default)."""
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        if mean.shape != (12,) or covariance.shape != (12, 12):
            raise ValueError("mean/covariance have to be shapes (12,) and (12,12)")
        motion_mat = np.eye(12, dtype=np.float64)
        motion_mat[:6, 6:] = np.eye(6, dtype=np.float64) * float(dt)
        # Covariance scales are magnitudes.  A signed/near-zero component must
        # not make the process noise collapse or change sign.
        scale_xyz = max(float(np.linalg.norm(mean[:3])), 1e-3)
        scale_rot = max(float(np.linalg.norm(mean[3:6])), 1e-3)
        # Constant-velocity process model.  The old implementation assumed
        # dt=1 frame and added a fixed diagonal noise, which made a 30 FPS
        # stream vastly noisier than a 60 FPS stream.  Scale the position /
        # velocity blocks with the physically correct dt powers while keeping
        # the existing relative noise weights as acceleration scales.
        acceleration_std = np.array(
            [
                self._std_weight_trans * scale_xyz,
                self._std_weight_trans * scale_xyz,
                self._std_weight_trans * scale_xyz,
                self._std_weight_rot * scale_rot,
                self._std_weight_rot * scale_rot,
                self._std_weight_rot * scale_rot,
            ],
            dtype=np.float64,
        )
        dt2 = float(dt) * float(dt)
        dt3 = dt2 * float(dt)
        dt4 = dt2 * dt2
        motion_cov = np.zeros((12, 12), dtype=np.float64)
        for index, variance in enumerate(np.square(acceleration_std)):
            motion_cov[index, index] = 0.25 * dt4 * variance
            motion_cov[index, 6 + index] = 0.5 * dt3 * variance
            motion_cov[6 + index, index] = 0.5 * dt3 * variance
            motion_cov[6 + index, 6 + index] = dt2 * variance
        mean = np.dot(mean, motion_mat.T)
        mean = self._normalize_state_angles(mean)
        # accumulate the variance
        covariance = np.linalg.multi_dot((
            motion_mat, covariance, motion_mat.T)) + motion_cov
        
        return mean, covariance

    def project(self, mean, covariance, measurement_covariance=None):
        """Project state to measurement space"""
        measurement_covariance = self._validate_measurement_covariance(
            measurement_covariance
        )
        # Initialize uncertainties
        scale_xyz = max(abs(float(mean[2])), 1e-3)
        scale_rot = max(float(np.linalg.norm(mean[3:6])), 1e-3)
        
        std = [
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
        ]
        innovation_cov = np.diag(np.square(std))
        if measurement_covariance is not None:
            innovation_cov += measurement_covariance
        
        # Project state
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        
        return projected_mean, projected_cov
    
    def project_for_xy(self, mean, covariance):
        # Image-space bbox centres are converted back to metric x/y using the
        # current depth.  Their uncertainty therefore scales with object
        # range, rather than with the distance from the optical axis.  The old
        # norm(mean[:2]) model became almost zero when the object was centred
        # in the image and made Cutie corrections unrealistically confident.
        scale_xy = max(float(np.linalg.norm(mean[:3])), 1e-3)
        
        # xy sensor noise model
        std_xy = [
            self._std_weight_trans_xy * scale_xy,
            self._std_weight_trans_xy * scale_xy
        ]
        innovation_cov = np.diag(np.square(std_xy))
        
        projected_mean = np.dot(self._update_mat_xy, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat_xy, covariance, self._update_mat_xy.T)) + innovation_cov
        
        return projected_mean, projected_cov

    def innovation_distance(
        self, mean, covariance, measurement, measurement_covariance=None
    ):
        """Return the squared Mahalanobis distance for a 6D measurement."""
        projected_mean, projected_cov = self.project(
            mean, covariance, measurement_covariance=measurement_covariance
        )
        innovation = self._pose_innovation(measurement, projected_mean)
        solved = scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False),
            innovation,
            check_finite=False,
        )
        return float(np.dot(innovation, solved))

    def update(
        self,
        mean,
        covariance,
        measurement,
        measurement_covariance=None,
        gate_threshold=None,
    ):
        """Kalman filter correction step"""
        measurement_covariance = self._validate_measurement_covariance(
            measurement_covariance
        )
        projected_mean, projected_cov = self.project(
            mean, covariance, measurement_covariance=measurement_covariance
        )
        
        # calculate kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        
        # Update with measurement
        innovation = self._pose_innovation(measurement, projected_mean)
        if gate_threshold is not None:
            if not np.isfinite(gate_threshold) or gate_threshold <= 0.0:
                raise ValueError("gate_threshold must be finite and positive")
            innovation_distance = float(
                np.dot(
                    innovation,
                    scipy.linalg.cho_solve(
                        (chol_factor, lower), innovation, check_finite=False
                    ),
                )
            )
            if innovation_distance > gate_threshold:
                return np.asarray(mean, dtype=np.float64), np.asarray(
                    covariance, dtype=np.float64
                )
        # reweight
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_mean = self._normalize_state_angles(new_mean)
        # Joseph covariance update.  The shorter ``P-KSK'`` expression is
        # mathematically equivalent, but repeated two-camera updates can make
        # it lose symmetry or produce a tiny negative eigenvalue in floating
        # point arithmetic.
        observation = self._update_mat
        if measurement_covariance is None:
            measurement_noise = projected_cov - np.linalg.multi_dot(
                (observation, covariance, observation.T)
            )
        else:
            measurement_noise = measurement_covariance
        identity = np.eye(2 * 6, dtype=np.float64)
        residual_map = identity - np.dot(kalman_gain, observation)
        new_covariance = (
            residual_map @ covariance @ residual_map.T
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )
        new_covariance = 0.5 * (new_covariance + new_covariance.T)
        # Keep the state numerically positive definite for the next Cholesky
        # projection without changing covariance values at normal scales.
        eigenvalues, eigenvectors = np.linalg.eigh(new_covariance)
        if eigenvalues.min() < 1e-12:
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            new_covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        
        return new_mean, new_covariance

    def update_many(
        self,
        mean,
        covariance,
        measurements,
        measurement_covariances=None,
        gate_threshold=None,
    ):
        """Sequentially fuse independent sensors into one shared state.

        Calling ``update`` once per camera is the standard Kalman formulation
        for independent measurements.  The second update sees the covariance
        reduced by the first update, so the sensors are not double-counted.
        ``accepted`` and ``innovation_distances`` are returned for diagnostics.
        """
        measurements = list(measurements)
        if not measurements:
            return mean, covariance, {
                "accepted": [],
                "innovation_distances": [],
            }
        if measurement_covariances is None:
            measurement_covariances = [None] * len(measurements)
        else:
            measurement_covariances = list(measurement_covariances)
            if len(measurement_covariances) != len(measurements):
                raise ValueError("one covariance is required for each measurement")

        accepted = []
        distances = []
        for measurement, covariance_i in zip(measurements, measurement_covariances):
            distance = self.innovation_distance(
                mean, covariance, measurement, covariance_i
            )
            distances.append(distance)
            mean, covariance = self.update(
                mean,
                covariance,
                measurement,
                measurement_covariance=covariance_i,
                gate_threshold=gate_threshold,
            )
            accepted.append(
                np.isfinite(distance)
                and (gate_threshold is None or distance <= gate_threshold)
            )
        return mean, covariance, {
            "accepted": accepted,
            "innovation_distances": distances,
        }
    
    def update_from_xy(self, mean, covariance, measurement_xy):
        """Kalman filter correction step for xy"""
        proj_mean, proj_cov = self.project_for_xy(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat_xy.T).T,
            check_finite=False).T
        
        # Update with measurement
        innovation = measurement_xy - proj_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, proj_cov, kalman_gain.T))
        
        return new_mean, new_covariance

    # def gating_distance(self, mean, covariance, measurements,
    #                     only_position=False, metric='maha'):
    #     """Compute gating distance between state and measurements"""
    #     projected_mean, projected_cov = self.project(mean, covariance)
        
    #     if only_position:
    #         # Use only translation components
    #         projected_mean = projected_mean[:3]
    #         projected_cov = projected_cov[:3, :3]
    #         measurements = measurements[:, :3]
        
    #     d = measurements - projected_mean
        
    #     if metric == 'maha':
    #         cholesky_factor = np.linalg.cholesky(projected_cov)
    #         z = scipy.linalg.solve_triangular(
    #             cholesky_factor, d.T, lower=True, 
    #             check_finite=False, overwrite_b=True)
    #         return np.sum(z * z, axis=0)
    #     else:
    #         return np.sum(d * d, axis=1)
