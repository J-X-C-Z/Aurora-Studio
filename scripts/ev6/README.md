# EV6 image parity comparator

Offline reference/candidate image comparison for EV6 records. It compares one
pair per `PF<number>|<backend>` key and never resizes, converts channels, or
silently downcasts bit depth.

## Requirements

- Python 3.10+
- `numpy` (tested with 2.2.6)
- `Pillow` (tested with 12.3.0)
- Optional: `opencv-python` (tested with 5.1.0) for 16-bit RGB/RGBA PNG

No models, no network, no vendor assets.

## Roster

`expected` is mandatory, so an absent pair is never counted as a pass.

```json
{
  "items": [
    {
      "id": "pf90002-cuda",
      "pf": "PF90002",
      "backend": "cuda",
      "reference": "ref/pf90002_cuda.png",
      "candidate": "cand/pf90002_cuda.png",
      "expected": true,
      "roi": [0.25, 0.25, 0.75, 0.75]
    }
  ]
}
```

Paths are resolved relative to the roster file. `pf` is normalised to `PF`
plus 1–9 digits; `backend` is normalised to a lowercase label of 1–32 ASCII
letters, digits or hyphens, starting with a letter or digit. `roi` is optional
and normalised (`0 <= x0 < x1 <= 1`, same for y); a ROI that maps to zero pixels
is an error.
Statuses: `pass`, `fail`, `missing`, `unsupported`, `error`. The summary keeps
`expected_missing` apart from `pass` and from `not_collected`. Duplicate `id`
or `pf|backend` keys are rejected before any image is opened.

## Bit depth and decoding

- 8-bit `L`, `LA`, `RGB`, `RGBA`: Pillow, samples kept as loaded.
- 16-bit grayscale (`I;16`): Pillow, uint16 values kept.
- 16-bit `RGB`/`RGBA` PNG: OpenCV `IMREAD_UNCHANGED`, converted explicitly
  from BGR/BGRA to RGB/RGBA; low bits and alpha are preserved. Without OpenCV
  these files are `unsupported`, never an 8-bit downcast.
- Refused: palette PNG, 16-bit gray+alpha, 16-bit TIFF that would downcast,
  and unsupported Pillow modes.
- Size/channel mismatch is an error; 8-bit vs 16-bit is a `bit-depth mismatch`
  error. `--ignore-alpha` is an explicit RGBA-only opt-in.

## Statistics and heatmaps

Whole-frame and ROI inside/outside statistics are accumulated in 256-row
bands with int32 diffs and int64 accumulators. Each channel reports `max_abs`,
signed mean, mean absolute error, signed variance, an absolute-difference
histogram (0..15 plus `>=16`), and a 16-bin reference histogram.

Heatmaps are 8x nearest-neighbour (`np.repeat`, no interpolation):

- `<= 350,000` source pixels: one full-frame 8x PNG;
- larger frame with a ROI: an 8x ROI crop (or bounded ROI tiles);
- larger frame without a ROI: only 512x512 source tiles containing a non-zero
  difference, named with their `x`/`y` coordinates and listed in the report;
- at most 256 tiles per item; truncation is recorded.

Full-frame statistics stay complete even when only tiles are written.

## Usage

From the repository root:

```bash
python scripts/ev6/ev6_compare.py selftest --out scripts/ev6/.ev6-scratch
python scripts/ev6/ev6_compare.py run --roster roster.json --out scripts/ev6/.ev6-scratch/report
python -m unittest discover -s scripts/ev6/tests -v
```

From the tool directory:

```bash
cd scripts/ev6
python ev6_compare.py selftest --out .ev6-scratch
python -m unittest discover -s tests -v
```

`selftest` writes its synthetic samples under `<out>/ev6-selftest/` and the
report under `<out>/ev6-selftest/report/`; `run` writes `report.json`,
`report.md`, `report.html`, and `heatmaps/` under `--out`. Each run needs a
fresh output directory (for `selftest`, a fresh `ev6-selftest/` child). If it
already exists, the command exits 2 without overwriting or deleting it. The
example output location is ignored by Git. Reports describe local input file
paths; keep generated reports and inputs out of commits.

The optional 24MP check (6000x4000, 8-bit, ~11 s; peak RSS not measured):

```bash
EV6_RUN_24MP=1 python -m unittest discover -s scripts/ev6/tests -v
```

On PowerShell: `$env:EV6_RUN_24MP='1'; python -m unittest discover -s scripts/ev6/tests -v`.

## Tests

Default: 26 discovered tests, 25 pass and 1 optional 24 MP test skips with
OpenCV installed. Setting `EV6_RUN_24MP=1` ran all 26 successfully. Without
OpenCV, optional colour-16 decode tests skip and the self-test reports those
pairs as unsupported. Coverage includes
identical pairs, one-channel off-by-one, u16 single-low-bit differences,
alpha/RGBA order, cv2 RGB order and low bits, no-cv2 unsupported, 8-vs-16
mismatch, size mismatch, empty ROI, ROI in/out split, missing expected,
duplicate keys, small/large heatmap bounds, and CLI exit codes.

## Boundaries

- Synthetic software checks only. This does not claim hardware parity or
  OEM/vendor `EXACT`.
- The 24MP smoke test is 8-bit; a 24MP 16-bit end-to-end decode was not run.
- 16-bit RGB/RGBA depends on OpenCV; 16-bit gray+alpha is refused.
- Variance uses the population formula `sum_sq / n - mean^2`; it is exact for
  the tested ranges but not claimed robust for extreme mean/variance ratios.
- This tool does not implement multi-item batch ingest, a cross-platform
  orchestrator, or a dashboard.
