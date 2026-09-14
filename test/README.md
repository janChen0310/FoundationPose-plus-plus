# Live RealSense pose tracking

Run the live Intel RealSense adapter with the Box mesh, Cutie 2D tracking, and
Kalman filtering:

```bash
conda activate pose
cd /home/pair1/hsieh/dr_nas/FoundationPose-plus-plus

python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

The adapter defaults to a `640 x 480` RGB-D stream at `30 FPS`. Press `s` to
select the object ROI, `r` to register it again, and `q` or `Esc` to quit.

## Resolution combinations

The color stream is selected with `--width` and `--height`. The depth stream
can be selected independently with `--depth-width` and `--depth-height`. The
script aligns depth to color before inference, so the depth array passed to
FoundationPose has the color image dimensions.

The commands below all use camera_0 (`254322070766`) and the same Box mesh.
They are common D435/D435i profiles; the exact profiles available depend on
the camera firmware and USB connection. Run `rs-enumerate-devices` if a
particular profile is rejected by `pipeline.start()`.

### 640 x 480 color + 640 x 480 depth, 30 FPS (default and lowest load)

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --width 640 --height 480 \
  --depth-width 640 --depth-height 480 \
  --fps 30 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

### 1280 x 720 color + 1280 x 720 depth, 30 FPS (balanced)

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --width 1280 --height 720 \
  --depth-width 1280 --depth-height 720 \
  --fps 30 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

### 1920 x 1080 color + 1280 x 720 depth, 30 FPS (highest color detail)

The D435 color sensor supports this high-resolution color profile while the
depth sensor normally uses `1280 x 720`. Alignment produces a `1920 x 1080`
depth image for FoundationPose.

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --width 1920 --height 1080 \
  --depth-width 1280 --depth-height 720 \
  --fps 30 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

### 848 x 480 color + 848 x 480 depth, 60 FPS (higher temporal rate)

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --width 848 --height 480 \
  --depth-width 848 --depth-height 480 \
  --fps 60 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

### 960 x 540 color + 848 x 480 depth, 30 FPS

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --width 960 --height 540 \
  --depth-width 848 --depth-height 480 \
  --fps 30 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

### 640 x 360 color + 640 x 360 depth, 60 FPS

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --width 640 --height 360 \
  --depth-width 640 --depth-height 360 \
  --fps 60 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

### 424 x 240 color + 424 x 240 depth, 60 FPS (lowest bandwidth)

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 254322070766 \
  --width 424 --height 240 \
  --depth-width 424 --depth-height 240 \
  --fps 60 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

Higher resolution increases depth-alignment, Cutie, OpenCV, and FoundationPose
load. Start with `1280 x 720 @ 30 FPS` for a balance of detail and real-time
performance; use `1920 x 1080` color when image detail is more important than
latency, or `848 x 480 @ 60 FPS` when temporal response is more important.

Use `--mesh-scale` if the STL units do not already match the physical object.
For an STL expressed in millimeters, use `--mesh-scale 0.001` to convert it to
meters.

## Single-camera Kalman behavior

`live_realsense_pose.py` intentionally uses the original FoundationPose++
single-view Kalman behavior so it matches the JHU reference for either camera.
The filter is a local prior, not a second sensor fusion result. For each image,
Cutie optionally corrects `x/y`, FoundationPose runs once, then the local state
predicts one processed-image step for the next call. This avoids the
timestamp-aware rig filter changing the prior seen by `camera_1`.

To test `camera_1` (`261722076377`) with the same RGB-D profile as the JHU
script, run:

```bash
python test/live_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --serial 261722076377 \
  --width 640 --height 480 \
  --depth-width 640 --depth-height 480 \
  --fps 30 \
  --activate-2d-tracker \
  --activate-kalman-filter
```

## Two-camera fusion

`src/utils/stereo_pose_fusion.py` contains the camera-independent fusion layer
used for dual-camera experiments. FoundationPose produces
`T_camera_object`; the caller supplies the calibrated `T_rig_camera` for each
camera. The layer then:

1. transforms both measurements to the rig frame;
2. rotates each 6x6 covariance into that frame;
3. rejects pairs whose acquisition timestamps exceed the configured skew;
4. updates one shared Kalman filter in timestamp order;
5. applies a Mahalanobis innovation gate to reject an inconsistent camera.

Do not run two filters and average their outputs. That loses the covariance
relationship and can hide a bad camera measurement. The production ROS path
already follows this same design in `motion_tracking`'s
`AcquisitionAwareSe3Filter`: a single world/rig state is updated once per
camera observation and published on `/rig/objects/cube_53p11/pose`.

The live adapter opens both RGB-D cameras directly and reads each camera's
intrinsics from its active stream profile. It uses the calibrated
`T_rig_camera_0` and `T_rig_camera_1` values only for the rigid transform
between each measurement and the fusion rig. Stop any `motion_tracking` process
that already owns either camera before running this command:

```bash
cd /home/pair1/hsieh/dr_nas/FoundationPose-plus-plus && \
conda run --no-capture-output -n pose python test/live_stereo_realsense_pose.py \
  --mesh test/mesh/Box_53mm.STL \
  --calibration /home/pair1/hsieh/dr_nas/motion_tracking/config/calibration/current.yaml \
  --serial0 254322070766 \
  --serial1 261722076377 \
  --width 640 --height 480 --fps 30 \
  --timestamp-source host \
  --max-skew-ms 50 \
  --serial-track \
  --keep-local-prior \
  --activate-2d-tracker \
  --activate-kalman-filter \
  --max-track-translation-jump-mm 10 \
  --max-track-rotation-jump-deg 5 \
  --max-track-speed-mm-s 1000 \
  --max-track-rotation-speed-deg-s 360 \
  --output test/stereo_pose_debug_serial.npz \
  --jsonl-output test/stereo_pose_debug_serial.jsonl
```

The command opens one ROI window per camera during initialization. Select the
same physical Box in both windows, then leave the object visible to both
cameras while the test runs. Press `q` or `Esc` to stop. Each camera has an
independent Cutie state and a local single-view FP++ prior. The shared rig
filter receives the unfiltered `track_one()` measurement from each camera, so
one local prior is never counted twice as a second sensor. The view draws raw
camera-0 in blue, raw camera-1 in red, the accepted fused pose in green, and a
held fused pose in orange after all observations are rejected.

### Live-view legend

| Display | Meaning |
| --- | --- |
| Blue box (`camera_0 raw`) | Current, unfused FoundationPose++ candidate from camera 0. It remains visible when a gate excludes it from fusion. |
| Red box (`camera_1 raw`) | Current, unfused FoundationPose++ candidate from camera 1. It remains visible when a gate excludes it from fusion. |
| Green box and XYZ axes (`fused`) | Shared rig-frame pose after at least one current camera measurement was accepted by the normal gate or the guarded reacquisition path. This is the tracking result. |
| Orange box and XYZ axes (`fused`) | Held last-known fused pose. Both current measurements failed or were rejected, so this is stale and must not be interpreted as a new observation. |
| Yellow `MEASURED` status | At least one measurement updated or reacquired the green fused pose in this image pair. |
| Yellow `STALE` status | No measurement updated the fused pose; the orange pose is being held while the tracker waits to recover. |
| `age`, `loop`, `track` | Display latency, complete loop duration, and the two serial FoundationPose++ tracking calls respectively, in milliseconds. |

## Recommended camera0 runtime for DR-NAS

The step-8 runtime uses camera 0 as the only FoundationPose++ pose source and
camera 1 only as a calibrated projection view:

The runtime state machine is:

1. Decode AprilTag 36h11 IDs in camera 0 and accept IDs 6--11 from
   `cube_53p11.yaml`.
2. Solve the semantic cube pose with PnP and reject excessive reprojection
   error.
3. Project the configured cube corners, dilate them into an automatic RGB-D
   ROI, and verify that the ROI contains depth.
4. Display the red ROI and `REGISTERING...`, then run the blocking
   FoundationPose registration. The default 10 refinement iterations can
   freeze the preview for a few seconds.
5. Initialize Cutie and the Kalman filter from the registered result, then use
   camera-0 FoundationPose++ for every tracking update. Tags are only checked
   for recovery/disagreement; they are not fused into the normal pose.
6. Publish the semantic cube pose and locked `wuji_palm` pose to DR-NAS. Apply
   the calibrated rig transform to the same camera-0 result for the camera-1
   visualization; camera 1 is not a pose source.
7. Render an RViz-like rig pane below both camera images. By default its table
   plane is the locked Wuji palm Tag face and display +Z is the Tag's outward
   normal. It contains the calibrated camera frustums, locked palm platform,
   live cube, and numeric world/relative translations and RPY angles.

```bash
cd /home/pair1/hsieh/dr_nas/motion_tracking
./scripts/track_foundationpose_cube.sh \
  --cube-config config/objects/cube_53p11.yaml \
  --calibration config/calibration/current.yaml \
  --locked-reference config/camera_tags/wuji_palm_id1_locked.yaml \
  --exposure-us 2500 \
  --gain 80 \
  --rgb-width 960 \
  --rgb-height 540 \
  --fps 30 \
  --intrinsics-source auto \
  --inference-scale 1.0 \
  --depth-width 848 \
  --depth-height 480 \
  --display-width 720 \
  --scene-height 0 \
  --scene-frame locked-tag \
  --est-refine-iter 5 \
  --track-refine-iter 2 \
  --reinit-translation-mm 80 \
  --reinit-rotation-deg 80 \
  --reinit-confirmations 5 \
  --max-invalid-frames 5 \
  --frame-timeout-ms 2000 \
  --max-camera-restarts 3
```

No manual ROI is needed. A configured cube tag (ID 6--11) creates the initial
mask, and tags are consulted again only to recognize loss and re-register.
Normal frame-to-frame pose output is FoundationPose++ from camera 0. The
locked palm and cube share the exact camera-0 acquisition timestamp and are
sent directly through the existing DR-NAS Unix socket protocol.
By default capture and display retain the calibrated 1920x1080 stream, while
`--inference-scale 0.5` runs Cutie/FoundationPose on a 960x540 RGB-D copy.
`--rgb-width`, `--rgb-height`, and `--fps` may override the actual capture
profile. With `--intrinsics-source auto`, a changed RGB size uses each active
RealSense color profile's K/distortion throughout Tag PnP, ROI, tracking, and
rendering; the calibrated inter-camera rig transforms are retained. For
example, use `--rgb-width 960 --rgb-height 540 --fps 30 --depth-width 848
--depth-height 480 --inference-scale 1.0` for true 960x540 capture. The runtime
converts D435i inverse Brown-Conrady coefficients to a fitted OpenCV forward
model instead of passing inverse coefficients to OpenCV directly.

| Combined-window display | Meaning |
| --- | --- |
| Left pane, green box (`FP++ camera0`) | The current cube result produced by camera-0 FoundationPose++. This is the pose sent to DR-NAS. |
| Right pane, yellow box (`camera1 projected`) | The camera-0 result transformed through the calibrated rig extrinsics and projected into camera 1. It is visualization only, not a second measurement. |
| Bottom 3-D pane | Full-width orbit scene with a 0.10 m grid, both calibrated camera poses/frustums, locked palm axes/platform, and the current cube pose. The default `--scene-frame locked-tag` makes the locked Wuji palm Tag face the table XY plane and its outward `+Z` normal point up. Use `--scene-frame world` for the raw rig orientation. `--scene-height 0` initially makes it equal in height to the camera row. |
| Bottom numeric overlay | A translucent upper-left overlay reports `camera_1 in camera_0`, locked palm in world, cube in world, and cube relative to locked palm; each pose includes translation and RPY. These values retain their original world/relative semantics even when the geometry uses the Tag-aligned display frame. |
| 3-D interaction | Put the pointer in the bottom pane: left-drag orbits, middle-drag pans, and the mouse wheel zooms only the 3-D content (0.35x--5.0x). `v` or `0` restores the camera-0-side default orbit, pan center, and 1.0x zoom; `r` remains tracking re-registration. The upper-right `3-D CONTROLS` legend lists every mouse/key binding. |
| Window resize | Drag the native window border. Both camera images preserve their aspect ratio and scale with their panes; the bottom scene is redrawn at the new client size. |
| Red translucent region / contour | The automatic RGB-D initialization ROI projected from the accepted cube Tag pose. |
| Red/green/blue axes | FoundationPose mesh-frame X/Y/Z axes. The bottom `t=[x,y,z] m` is the semantic cube translation in that camera frame. |
| `ROI READY - FoundationPose REGISTERING...` | The Tag and ROI gates passed. Registration is running synchronously and the displayed camera frame will pause briefly. |
| Cyan marker borders / `raw tags=[...]` | Every AprilTag decoded in the camera-0 image. An empty list points to image quality, exposure, scale, or occlusion rather than FoundationPose. |
| `cube tags=[...]` / `expected=[...]` | Decoded IDs accepted by `cube_53p11.yaml`, and the configured IDs. Raw IDs present but no cube IDs means the printed/configured IDs do not match. |
| `waiting_for_cube_tags` | No configured cube tag has produced an automatic initialization pose yet. |
| `registered_from_cube_tags` | Tags supplied the ROI and semantic cube-frame alignment for registration/re-registration. |
| `fast_reseed_from_cube_tags` | A post-registration recovery used the known mesh-to-cube bridge to reset FoundationPose, Cutie, and Kalman directly; no 252-candidate global search was repeated. |
| `foundationpose_camera0` | Normal tracking; tags are not fused into this result. |
| `camera1 stream: STALLED (tracking unaffected)` | The projection pane is retaining its last image; camera-0 tracking and IPC continue without waiting for camera 1. |
| `IPC timestamp: NOT AUTHORITATIVE` | The RealSense frame lacks a valid global exposure-midpoint timestamp, so the fail-closed runtime does not publish a usable cube measurement. |
| `DR-NAS IPC: waiting for consumer` | The downstream process has not bound the configured Unix datagram socket yet; visualization continues, but packets are being dropped. |
| `age` / `loop` | Acquisition-to-display age and one runtime iteration. Red `age` exceeds the current downstream 67 ms stale gate. |

If camera 0 produces no frame for `--frame-timeout-ms` (default 2000 ms), the
runtime releases and restarts both RealSense pipelines, invalidates the old pose,
and waits for the cube tags before registering again. It exits after
`--max-camera-restarts` attempts (default 3). The two configured D435i devices
must preferably be connected to different USB root ports; sharing one external
hub can still cause transient starvation even when both report USB 3.2.

The initial red ROI is followed by one unavoidable global registration. Later
tag disagreement or tracking loss uses a fast Tag reseed once `T_mesh_cube` is
known. A thin red line extending from the pose origin is the X axis; only the
translucent red fill and enclosing contour denote the ROI. Verbose third-party
per-frame INFO tracing is suppressed to keep terminal I/O out of the real-time
loop.

Press `r` to discard the current track and wait for tag-based re-registration;
press `q` or `Esc` to stop. The STL mesh frame is bridged to the semantic
`cube_53p11.yaml` object frame at first registration, so downstream control
does not receive the STL corner-origin frame.

### Stereo experiment behavior

The following notes apply to `live_stereo_realsense_pose.py`, not to the
recommended camera0-only runtime above.

For a geometrically symmetric cube, several object-frame rotations describe
the same visible box. Camera 0 therefore selects the cube-equivalent rotation
nearest its preceding local observation, and camera 1 selects the branch
nearest camera 0's current rig-frame branch. The raw jump history advances on
each locally valid observation independently of the shared fusion decision, so
rejecting camera 1 once cannot permanently freeze its comparison reference. If
a jump is rejected, two consecutive candidates that are mutually consistent
relock that camera's local jump reference. Rejected candidates are still drawn
as raw blue/red boxes for diagnosis; rejection only excludes them from the
green fused result. If a sudden motion makes the temporal filter reject
everything, two consecutive
camera pairs that agree within 20 mm and 15 degrees automatically reinitialize
the shared filter. When the cameras do not agree, the measurement with the
smaller innovation is used as a temporary single-camera anchor; the other
camera remains rejected until it becomes consistent again. The existing raw
jump gate runs before this fallback, so an implausible FoundationPose jump
cannot become an anchor. This prevents the green/orange result from remaining
locked to an obsolete pose while a raw box is tracking the cube correctly. The
stereo-consensus limits can be changed with `--reacquire-after-rejections`,
`--reacquire-translation-mm`, and `--reacquire-rotation-deg`.

Two diagnostic files are written while the process is running:

* `test/stereo_pose_debug_serial.jsonl` contains one JSON object per image pair. Each
  record includes both frame numbers, host/device/capture timestamps, each
  camera's `T_camera_object`, each converted `T_rig_object`, the fused pose,
  accepted/rejected sources, innovation distances, tracking/fusion/draw
  timings, and display age.
* `test/stereo_pose_debug_serial.npz` contains the same records plus numeric arrays
  (`camera_0_t_rig_object`, `camera_1_t_rig_object`, `fused_t_rig_object`,
  `capture_timestamps_ns`, `innovation_distances`, and timing arrays) for
  offline plotting. Use `--jsonl-output ""` to disable the text log.

Stale RealSense frames are drained before each pair. The command uses
`--serial-track`, which is required when the FP++ options are active: the two
Cutie/CUDA/nvdiffrast states are independent and must not run concurrently in
threads. The 53 mm Box mesh is treated as a cube in both base and FP++ modes;
the code canonicalizes its 24 equivalent rotational branches against the prior
before comparing the two cameras. A per-camera jump gate uses 10 mm / 5 degrees
as a noise floor and then scales the limit with elapsed image time. If the
normal innovation gate rejects both cameras, the guarded recovery path above
either reacquires from current evidence or holds the last pose and clears its
velocity when no safe candidate exists. A rejected camera usually points to
its ROI, `T_rig_camera_1`, mesh frame, depth quality, or timestamp pairing.
JSONL contains both canonicalized and raw jump metrics, the local Cutie box and
timing, camera measurements, fusion decisions, reacquisition mode, and display
age for later debugging.
