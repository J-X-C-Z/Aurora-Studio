"""Unit and end-to-end tests for the standalone EV6 image comparator.

Test suite map (all fast, deterministic, synthetic; no network, no vendor files):
  * unit: identical / one-channel-off-by-one / alpha / channel mismatch /
    size mismatch / empty ROI / ROI in-out split / 16-bit gray / 16-bit RGB
    refusal / heatmap 8x nearest / roster validation;
  * end-to-end: `selftest` report + one CLI exit-code matrix.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ev6_compare as ev6  # noqa: E402


def rgb_frame(height: int = 8, width: int = 8, value: int = 100) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = value
    frame[..., 1] = value // 2
    frame[..., 2] = value // 3
    return frame


def rgba_frame(height: int = 8, width: int = 8, value: int = 100) -> np.ndarray:
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb_frame(height, width, value), alpha], axis=2)


def save(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


class CompareArraysTests(unittest.TestCase):
    def test_identical_rgb_is_bit_exact_pass(self):
        reference = rgb_frame()
        result = ev6.compare_arrays(reference, reference.copy(), ["R", "G", "B"], None, 0, False)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["bit_exact"])
        self.assertEqual(result["whole"]["max_abs"], 0)

    def test_one_channel_off_by_one_fails_at_zero_tolerance(self):
        reference = rgb_frame()
        candidate = reference.copy()
        candidate[2, 3, 0] = np.uint8(reference[2, 3, 0] + 1)
        result = ev6.compare_arrays(reference, candidate, ["R", "G", "B"], None, 0, False)
        self.assertEqual(result["status"], "fail")
        channels = {channel["name"]: channel for channel in result["whole"]["channels"]}
        self.assertEqual(channels["R"]["max_abs"], 1)
        self.assertEqual(channels["G"]["max_abs"], 0)
        self.assertEqual(channels["B"]["max_abs"], 0)
        tolerant = ev6.compare_arrays(reference, candidate, ["R", "G", "B"], None, 1, False)
        self.assertEqual(tolerant["status"], "pass")

    def test_alpha_difference_fails_and_explicit_ignore_alpha_passes(self):
        reference = rgba_frame()
        candidate = reference.copy()
        candidate[1, 1, 3] = 0
        strict = ev6.compare_arrays(reference, candidate, ["R", "G", "B", "A"], None, 0, False)
        self.assertEqual(strict["status"], "fail")
        self.assertEqual(strict["channels"], 4)
        ignored = ev6.compare_arrays(reference, candidate, ["R", "G", "B", "A"], None, 0, True)
        self.assertEqual(ignored["status"], "pass")
        self.assertEqual(ignored["channels"], 3)

    def test_rgb_vs_rgba_is_an_error_not_a_conversion(self):
        with self.assertRaises(ev6.ComparisonError) as caught:
            ev6.compare_arrays(rgb_frame(), rgba_frame(), ["R", "G", "B"], None, 0, False)
        self.assertIn("size/channel mismatch", str(caught.exception))

    def test_size_mismatch_is_an_error_without_resize(self):
        with self.assertRaises(ev6.ComparisonError) as caught:
            ev6.compare_arrays(rgb_frame(8, 8), rgb_frame(4, 4), ["R", "G", "B"], None, 0, False)
        message = str(caught.exception)
        self.assertIn("(8, 8, 3)", message)
        self.assertIn("(4, 4, 3)", message)
        self.assertIn("no resize", message)

    def test_empty_roi_raises(self):
        with self.assertRaises(ev6.EmptyRoiError):
            ev6.parse_roi([0.5, 0.5, 0.5, 0.5], 8, 8)
        with self.assertRaises(ev6.EmptyRoiError):
            ev6.parse_roi([0.0, 0.0, 1.1, 1.0], 8, 8)
        with self.assertRaises(ev6.EmptyRoiError):
            ev6.parse_roi([0.4, 0.4, 0.4, 0.9], 8, 8)

    def test_roi_inside_and_outside_are_separated(self):
        reference = np.zeros((16, 16, 3), dtype=np.uint8)
        candidate = reference.copy()
        candidate[4:12, 4:12, 0] = 1
        result = ev6.compare_arrays(reference, candidate, ["R", "G", "B"], [0.25, 0.25, 0.75, 0.75], 0, False)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["roi_pixels"], [4, 4, 12, 12])
        self.assertEqual(result["roi_inside"]["max_abs"], 1)
        self.assertEqual(result["roi_outside"]["max_abs"], 0)
        self.assertEqual(result["whole"]["max_abs"], 1)
        self.assertEqual(result["roi_inside"]["pixels"], 64)


class LoaderTests(unittest.TestCase):
    def test_16bit_grayscale_keeps_uint16_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gray16.png"
            values = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000)
            Image.fromarray(values).save(path)
            loaded, meta = ev6.load_image(path)
            self.assertEqual(loaded.dtype, np.uint16)
            self.assertEqual(meta.bit_depth, 16)
            self.assertEqual(int(loaded.max()), 63000)
            identical = ev6.compare_arrays(loaded, loaded.copy(), ["Y"], None, 0, False)
            self.assertEqual(identical["status"], "pass")
            changed = loaded.copy()
            changed[0, 0] = np.uint16(changed[0, 0] + 1)
            diff = ev6.compare_arrays(loaded, changed, ["Y"], None, 0, False)
            self.assertEqual(diff["whole"]["max_abs"], 1)

    def test_16bit_rgb_png_without_cv2_is_explicitly_unsupported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rgb16.png"
            ev6._png16_rgb(path, np.zeros((4, 4, 3), dtype=np.uint16))
            saved = (ev6.cv2, ev6.CV2_AVAILABLE)
            ev6.cv2 = None
            ev6.CV2_AVAILABLE = False
            try:
                with self.assertRaises(ev6.UnsupportedImage) as caught:
                    ev6.probe_image(path)
                self.assertIn("OpenCV", str(caught.exception))
            finally:
                ev6.cv2, ev6.CV2_AVAILABLE = saved

    @unittest.skipUnless(ev6.CV2_AVAILABLE, "OpenCV is not installed")
    def test_16bit_rgb_png_decodes_rgb_order_and_low_bits(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rgb16.png"
            values = ev6._rgb16_pattern(8, 6)
            ev6._png16_rgb(path, values)
            loaded, meta = ev6.load_image(path)
            self.assertEqual(meta.bit_depth, 16)
            self.assertEqual(meta.channels, 3)
            self.assertEqual(loaded.dtype, np.uint16)
            self.assertEqual(loaded[0, 0].tolist(), [0x0001, 0x8000, 0xFFFF])
            self.assertEqual(loaded[1, 1].tolist(), [0x1234, 0x2345, 0x3456])
            np.testing.assert_array_equal(loaded, values)

    @unittest.skipUnless(ev6.CV2_AVAILABLE, "OpenCV is not installed")
    def test_16bit_rgba_png_keeps_alpha_and_rgba_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "ref.png"
            candidate = root / "cand.png"
            values = ev6._rgba16_pattern(8, 6)
            changed = values.copy()
            changed[2, 3, 3] = 0x0001
            ev6._png16_rgba(reference, values)
            ev6._png16_rgba(candidate, changed)
            loaded, meta = ev6.load_image(reference)
            self.assertEqual(meta.channels, 4)
            self.assertEqual(loaded[0, 0].tolist(), [0x0001, 0x8000, 0xFFFF, 0xFFFE])
            self.assertEqual(loaded[2, 3].tolist(), [0x1234, 0x2345, 0x3456, 0xFFFE])
            strict = ev6.compare_arrays(loaded, ev6.load_image(candidate)[0], ["R", "G", "B", "A"], None, 0, False)
            self.assertEqual(strict["status"], "fail")
            ignored = ev6.compare_arrays(loaded, ev6.load_image(candidate)[0], ["R", "G", "B", "A"], None, 0, True)
            self.assertEqual(ignored["status"], "pass")

    @unittest.skipUnless(ev6.CV2_AVAILABLE, "OpenCV is not installed")
    def test_u16_single_low_bit_difference_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "ref.png"
            candidate = root / "cand.png"
            values = np.full((6, 8, 3), 0x1000, dtype=np.uint16)
            changed = values.copy()
            changed[3, 4, 0] = 0x1001
            ev6._png16_rgb(reference, values)
            ev6._png16_rgb(candidate, changed)
            ref_array, _ = ev6.load_image(reference)
            cand_array, _ = ev6.load_image(candidate)
            self.assertEqual(int(ref_array[3, 4, 0]), 0x1000)
            self.assertEqual(int(cand_array[3, 4, 0]), 0x1001)
            strict = ev6.compare_arrays(ref_array, cand_array, ["R", "G", "B"], None, 0, False)
            self.assertEqual(strict["status"], "fail")
            self.assertEqual(strict["whole"]["channels"][0]["max_abs"], 1)
            tolerant = ev6.compare_arrays(ref_array, cand_array, ["R", "G", "B"], None, 1, False)
            self.assertEqual(tolerant["status"], "pass")

    def test_8bit_and_16bit_are_never_compared_implicitly(self):
        with self.assertRaises(ev6.ComparisonError) as caught:
            ev6.compare_arrays(
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint16),
                ["R", "G", "B"], None, 0, False,
            )
        self.assertIn("bit-depth mismatch", str(caught.exception))


class HeatmapTests(unittest.TestCase):
    def test_heatmap_is_eight_times_nearest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "heat.png"
            absdiff = np.zeros((3, 4), dtype=np.int64)
            absdiff[1, 2] = 255
            ev6.save_heatmap(absdiff, path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (32, 24))
                pixels = np.array(image)
            hot = pixels[8:16, 16:24]  # 8x8 block of the single hot pixel
            self.assertTrue((hot == hot[0, 0]).all())
            cold = pixels[8:16, 8:16]
            self.assertFalse((cold == hot).all())
            self.assertGreater(int(hot[0, 0, 0]), int(cold[0, 0, 0]))


class RosterTests(unittest.TestCase):
    def _roster(self, directory: Path, items: list[dict]) -> Path:
        path = directory / "roster.json"
        path.write_text(json.dumps({"items": items}), encoding="utf-8")
        return path

    def test_missing_expected_is_missing_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / "ref.png", rgb_frame())
            roster = self._roster(root, [{
                "id": "missing", "pf": "PF90085", "backend": "wgsl",
                "reference": "ref.png", "candidate": "absent.png", "expected": True,
            }])
            summary = ev6.run_roster(roster, root / "out", 0, False)
            self.assertEqual(summary["pass"], 0)
            self.assertEqual(summary["expected_missing"], 1)
            self.assertFalse(summary["all_expected_pass"])
            report = json.loads((root / "out" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["items"][0]["status"], "missing")

    def test_not_collected_is_kept_apart_from_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / "ref.png", rgb_frame())
            save(root / "cand.png", rgb_frame())
            roster = self._roster(root, [
                {"id": "pass", "pf": "PF90002", "backend": "cuda", "reference": "ref.png", "candidate": "cand.png", "expected": True},
                {"id": "unseen", "pf": "PF101", "backend": "metal", "reference": "ref.png", "candidate": "absent.png", "expected": False},
            ])
            summary = ev6.run_roster(roster, root / "out", 0, False)
            self.assertEqual(summary["pass"], 1)
            self.assertEqual(summary["not_collected"], 1)
            self.assertEqual(summary["expected_missing"], 0)
            self.assertTrue(summary["all_expected_pass"])

    def test_duplicate_id_or_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            duplicate_id = self._roster(root, [
                {"id": "same", "pf": "PF1", "backend": "cuda", "reference": "a", "candidate": "b", "expected": True},
                {"id": "same", "pf": "PF2", "backend": "cuda", "reference": "a", "candidate": "b", "expected": True},
            ])
            with self.assertRaises(ev6.RosterError) as caught:
                ev6.load_roster(duplicate_id)
            self.assertIn("duplicate id", str(caught.exception))
            duplicate_key = self._roster(root, [
                {"id": "one", "pf": "PF1", "backend": "cuda", "reference": "a", "candidate": "b", "expected": True},
                {"id": "two", "pf": "PF1", "backend": "CUDA", "reference": "a", "candidate": "b", "expected": True},
            ])
            with self.assertRaises(ev6.RosterError) as caught:
                ev6.load_roster(duplicate_key)
            self.assertIn("duplicate key", str(caught.exception))

    def test_roster_requires_explicit_expected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roster = self._roster(root, [
                {"id": "x", "pf": "PF1", "backend": "cuda", "reference": "a", "candidate": "b"},
            ])
            with self.assertRaises(ev6.RosterError) as caught:
                ev6.load_roster(roster)
            self.assertIn("expected", str(caught.exception))


class HeatmapBudgetTests(unittest.TestCase):
    def test_small_frame_gets_one_full_frame_heatmap(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            reference = np.zeros((48, 64, 3), dtype=np.uint8)
            results, truncated = ev6.write_heatmaps(reference, reference.copy(), None, output, "PF1|cuda", False)
            self.assertFalse(truncated)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["x"], 0)
            self.assertEqual(results[0]["y"], 0)
            self.assertEqual(results[0]["width"], 64)
            self.assertEqual(results[0]["height"], 48)
            with Image.open(output / results[0]["path"]) as image:
                self.assertEqual(image.size, (64 * 8, 48 * 8))

    def test_large_frame_uses_tiles_with_explicit_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            reference = np.zeros((700, 700, 3), dtype=np.uint8)
            candidate = reference.copy()
            candidate[600:700, 600:700, 0] = 1
            results, truncated = ev6.write_heatmaps(reference, candidate, None, output, "PF2|cuda", False)
            self.assertFalse(truncated)
            self.assertEqual(len(results), 1)
            self.assertEqual((results[0]["x"], results[0]["y"]), (512, 512))
            self.assertEqual((results[0]["width"], results[0]["height"]), (188, 188))
            self.assertEqual(results[0]["max_abs"], 1)
            with Image.open(output / results[0]["path"]) as image:
                self.assertEqual(image.size, (188 * 8, 188 * 8))

    def test_large_frame_roi_crop_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            reference = np.zeros((700, 700, 3), dtype=np.uint8)
            results, truncated = ev6.write_heatmaps(
                reference, reference.copy(), [0.5, 0.5, 0.6, 0.6], output, "PF3|cuda", False
            )
            self.assertFalse(truncated)
            self.assertEqual(len(results), 1)
            self.assertEqual((results[0]["x"], results[0]["y"]), (350, 350))
            self.assertEqual((results[0]["width"], results[0]["height"]), (70, 70))
            with Image.open(output / results[0]["path"]) as image:
                self.assertEqual(image.size, (70 * 8, 70 * 8))

    @unittest.skipUnless(os.environ.get("EV6_RUN_24MP") == "1", "set EV6_RUN_24MP=1 to run the 24MP bounded check")
    def test_24mp_statistics_and_tiled_heatmap_stay_bounded(self):
        height, width = 4000, 6000
        reference = np.zeros((height, width, 3), dtype=np.uint8)
        candidate = reference.copy()
        candidate[800:1000, 1000:1300, 0] = 1
        result = ev6.compare_arrays(reference, candidate, ["R", "G", "B"], None, 0, False)
        self.assertEqual(result["whole"]["pixels"], height * width)
        self.assertEqual(result["whole"]["max_abs"], 1)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            results, truncated = ev6.write_heatmaps(reference, candidate, None, output, "PF24|cuda", False)
            self.assertFalse(truncated)
            self.assertGreaterEqual(len(results), 1)
            self.assertLessEqual(len(results), 4)
            self.assertTrue(all(tile["width"] <= ev6.HEATMAP_TILE_SOURCE for tile in results))
            self.assertTrue(all(tile["height"] <= ev6.HEATMAP_TILE_SOURCE for tile in results))
            self.assertTrue(all("_x" in tile["path"] for tile in results))


class EndToEndTests(unittest.TestCase):
    def test_existing_output_is_left_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "ev6-selftest"
            target.mkdir()
            sentinel = target / "report.json"
            sentinel.write_bytes(b"keep the existing output")
            roster = root / "roster.json"
            roster.write_text(json.dumps({"items": [{
                "pf": "PF1", "backend": "cuda", "expected": True,
                "reference": "ref.png", "candidate": "cand.png",
            }]}), encoding="utf-8")
            self.assertEqual(ev6.main(["selftest", "--out", str(root)]), 2)
            self.assertEqual(ev6.main(["run", "--roster", str(roster), "--out", str(target)]), 2)
            self.assertEqual(sentinel.read_bytes(), b"keep the existing output")
            self.assertEqual(list(target.iterdir()), [sentinel])

    def test_roster_key_validates_contract(self):
        self.assertEqual(ev6.roster_key(" pf90021 ", "Vulkan-FP32"), "PF90021|vulkan-fp32")
        for pf, backend in [("90021", "cuda"), (None, "cuda"), ("PF1", "cuda debug"), ("PF1", "")]:
            with self.subTest(pf=pf, backend=backend), self.assertRaises(ev6.RosterError):
                ev6.roster_key(pf, backend)

    def test_selftest_writes_offline_report_and_matches_designed_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "samples"
            summary = ev6.selftest(root)
            for key, value in ev6.selftest_expected().items():
                self.assertEqual(summary[key], value, key)
            self.assertTrue(summary["selftest_ok"])
            report_dir = root / "ev6-selftest" / "report"
            self.assertTrue((report_dir / "report.json").is_file())
            self.assertTrue((report_dir / "report.md").is_file())
            self.assertTrue((report_dir / "report.html").is_file())
            markdown = (report_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("PF90002|cuda", markdown)
            self.assertIn("hardware or OEM/vendor parity claim", markdown)
            self.assertTrue(any((report_dir / "heatmaps").glob("*.png")))
            measured = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(measured["summary"]["pass"], ev6.selftest_expected()["pass"])

    def test_cli_exit_codes_clean_roster_pass_fail_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / "ref.png", rgb_frame())
            save(root / "cand.png", rgb_frame())
            clean = root / "clean.json"
            clean.write_text(json.dumps({"items": [
                {"id": "ok", "pf": "PF1", "backend": "cuda", "reference": "ref.png", "candidate": "cand.png", "expected": True},
            ]}), encoding="utf-8")
            command = [sys.executable, str(ROOT / "ev6_compare.py"), "run", "--roster", str(clean), "--out", str(root / "out")]
            completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((root / "out" / "report.json").is_file())

            missing = root / "missing.json"
            missing.write_text(json.dumps({"items": [
                {"id": "gone", "pf": "PF2", "backend": "cuda", "reference": "ref.png", "candidate": "absent.png", "expected": True},
            ]}), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "ev6_compare.py"), "run", "--roster", str(missing), "--out", str(root / "out2")],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)

            duplicate = root / "duplicate.json"
            duplicate.write_text(json.dumps({"items": [
                {"id": "same", "pf": "PF3", "backend": "cuda", "reference": "ref.png", "candidate": "cand.png", "expected": True},
                {"id": "same", "pf": "PF4", "backend": "cuda", "reference": "ref.png", "candidate": "cand.png", "expected": True},
            ]}), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "ev6_compare.py"), "run", "--roster", str(duplicate), "--out", str(root / "out3")],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("duplicate id", completed.stderr)


if __name__ == "__main__":
    unittest.main()
