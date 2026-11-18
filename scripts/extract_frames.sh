#!/usr/bin/env bash
set -e
cd <workspace>
V=outputs/videos/model_49999/rl-video-step-0.mp4
mkdir -p outputs/videos/model_49999/frames
for t in 0.4 2.0 4.0 5.8; do
  ffmpeg -y -loglevel error -ss "$t" -i "$V" -frames:v 1 "outputs/videos/model_49999/frames/t${t}.png"
done
ls -la outputs/videos/model_49999/frames
ffprobe -v error -show_entries stream=width,height,nb_frames,r_frame_rate -of default=nw=1 "$V"
