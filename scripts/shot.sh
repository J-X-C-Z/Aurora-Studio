#!/usr/bin/env bash
# Runs the app on a virtual X display and saves a screenshot.
# usage: scripts/shot.sh out.png [seconds] [xdotool commands file]
set -u
OUT=${1:-/tmp/aurora.png}; WAIT=${2:-6}; ACTIONS=${3:-}
BIN=${BIN:-$(dirname "$0")/../target/debug/aurora-studio}
DISP=:77
Xvfb $DISP -screen 0 1720x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
XPID=$!
sleep 1
env -u WAYLAND_DISPLAY DISPLAY=$DISP VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  RUST_LOG=${RUST_LOG:-warn} "$BIN" >/tmp/aurora-run.log 2>&1 &
APID=$!
sleep "$WAIT"
if [ -n "$ACTIONS" ]; then DISPLAY=$DISP bash "$ACTIONS"; fi
DISPLAY=$DISP xwd -root -silent | convert xwd:- "$OUT"
kill $APID 2>/dev/null; sleep 0.3; kill -9 $APID 2>/dev/null; kill $XPID 2>/dev/null
echo "saved $OUT"; tail -5 /tmp/aurora-run.log
