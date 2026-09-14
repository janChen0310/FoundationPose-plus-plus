# Vendored FoundationPose++ provenance

This source snapshot is based on
[`Jianshu-Hu/FoundationPose-plus-plus`](https://github.com/Jianshu-Hu/FoundationPose-plus-plus)
commit `6717416` (`master`), with the DR-NAS RealSense, stereo diagnostics, and
camera0 monocular rollout integration applied locally.

Model checkpoints, downloaded weights, generated build trees, and live test
captures are intentionally excluded from the DR-NAS repository. Follow the
upstream installation instructions and the commands in `test/README.md` to
populate the local weight directories.

The optional Qwen2-VL and SAM-HQ demonstration stacks are also excluded because
the DR-NAS cube-tracking runtime uses FoundationPose and Cutie directly.
