# Pose model

Not committed — it's a 9.4 MB binary and easy to fetch. `swingcoach.pose` looks for
it at `models/pose_landmarker_full.task` by default (override with `--model`).

```
curl -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

`full` was used for the M0 spike and tracked well through motion blur at impact —
see `MISTAKES.md`. `lite` is faster and worth trying if per-frame latency matters
later; `heavy` is the accuracy ceiling if `full` turns out not to be enough.
