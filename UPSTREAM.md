# Provenance of the `drnas` branch

`drnas` is [`Jianshu-Hu/FoundationPose-plus-plus`](https://github.com/Jianshu-Hu/FoundationPose-plus-plus)
`master` (`6717416`) with the library changes that the DR-NAS motion-tracking node needs:

- `FoundationPose/estimater.py`: the mesh is exported into a private temporary directory per estimator instead of
  a shared `/tmp` file, so concurrent runs, or runs under another user, do not collide on its sidecar files.
- `src/utils/kalman_filter_6d.py`: the original single-camera filter is `LegacyKalmanFilter6D` (used by
  `src/obj_pose_track.py` and by DR-NAS); `KalmanFilter6D` is a time-aware filter beside it.
- `Cutie/cutie/model/utils/resnet.py`, `Cutie/cutie/utils/get_default_model.py`: checkpoints load on the CPU
  (FoundationPose makes CUDA the default tensor type), and Hydra's state is released after each model, so a
  process can build more than one Cutie tracker.
- `test/mesh/Box_53mm.STL`: the 53 mm cube.

DR-NAS checks this branch out as its git submodule `FoundationPose-plus-plus` and runs it from its own ROS node
(`motion_tracking/ros2_ws/src/mt_fpp`); `motion_tracking/scripts/build_fpp_env.sh` there builds the Python
environment and downloads the weights.  The live RealSense scripts that were in `test/`
(`live_realsense_pose.py`, `live_stereo_realsense_pose.py`, `live_monocular_cube_runtime.py`), their README and
`src/utils/stereo_pose_fusion.py` were removed on 2026-10-29, once the node had taken over their logic; the last
commit with them is `4c714b7`.
