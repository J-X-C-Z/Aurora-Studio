#!/usr/bin/env bash
# Starts the app on a virtual display, runs steps, saves screenshots.
# usage: scripts/tour.sh OUTDIR "name:keys" ...   keys are xdotool key names, '+' separated sequences by ' '
set -u
OUT=$1; shift; mkdir -p "$OUT"
BIN=${BIN:-$(dirname "$0")/../target/debug/aurora-studio}
DISP=:78
Xvfb $DISP -screen 0 1600x1000x24 -nolisten tcp >/tmp/xvfb2.log 2>&1 & XPID=$!
sleep 1
env -u WAYLAND_DISPLAY DISPLAY=$DISP VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json RUST_LOG=${RUST_LOG:-error} \
  "$BIN" ${APP_ARGS:-} >/tmp/aurora-tour.log 2>&1 & APID=$!
sleep ${WAIT:-7}
export DISPLAY=$DISP
WID=$(xdotool search --pid $APID 2>/dev/null | tail -1)
[ -n "$WID" ] && xdotool windowmove $WID 0 0 windowsize $WID 1600 1000 && sleep 1
xdotool mousemove 800 500 click 1; sleep 0.3
for step in "$@"; do
  name=${step%%:*}; cmds=${step#*:}
  for c in $cmds; do
    case $c in
      sleep=*) sleep ${c#sleep=};;
      click=*) xy=${c#click=}; xdotool mousemove ${xy%,*} ${xy#*,} click 1; sleep 0.2;;
      dclick=*) xy=${c#dclick=}; xdotool mousemove ${xy%,*} ${xy#*,} click --repeat 2 --delay 80 1; sleep 0.2;;
      drag=*) v=${c#drag=}; IFS=, read x1 y1 x2 y2 <<<"$v"; xdotool mousemove $x1 $y1 mousedown 1; for i in 1 2 3 4 5 6 7 8; do xdotool mousemove $((x1+(x2-x1)*i/8)) $((y1+(y2-y1)*i/8)); sleep 0.05; done; xdotool mouseup 1; sleep 0.3;;
      move=*) xy=${c#move=}; xdotool mousemove ${xy%,*} ${xy#*,}; sleep 0.2;;
      *) xdotool key $c; sleep 0.25;;
    esac
  done
  sleep ${SETTLE:-1.2}
  xwd -root -silent | convert xwd:- -crop 1600x1000+0+0 "$OUT/$name.png"
  echo "shot $name"
done
kill $APID 2>/dev/null; sleep 0.3; kill -9 $APID 2>/dev/null; kill $XPID 2>/dev/null
grep -v -E 'DRI3|xinput|SystemUIFont' /tmp/aurora-tour.log | tail -5
