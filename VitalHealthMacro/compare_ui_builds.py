#!/usr/bin/env python3
"""
compare_ui_builds.py — Automated Visual Regression, Masking, and Layout Fitting Tool.
Runs Maestro integration tests, captures screenshots, and compares them against benchmarks.
Supports screen exclusion, element masking (ignored regions), and out-of-bounds overflow checks.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# --- Dependency check and auto-installer ---
try:
    from PIL import Image, ImageChops, ImageDraw
except ImportError:
    print("PIL (Pillow) library is required for image processing. Installing it now...")
    subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"], check=True)
    from PIL import Image, ImageChops, ImageDraw


# --- Paths Configuration ---
SCRIPT_DIR = Path(__file__).parent
MAESTRO_DIR = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR.parent

VISUAL_DIR = MAESTRO_DIR / "visual_regression"
BENCHMARK_DIR = VISUAL_DIR / "benchmarks"
CURRENT_DIR = VISUAL_DIR / "current_run"
DIFF_DIR = VISUAL_DIR / "diffs"

# Expected screenshot names
SCREENSHOTS = [
    "01_welcome", "02_signin", "03_dashboard", "04_step_intelligence",
    "05_heart_scanner", "06_spo2_scanner", "07_calorie_intelligence",
    "08_medication_vault", "09_add_medicine", "10_symptom_signals",
    "11_digital_twin", "12_insights", "13_vault", "14_dr_aria", "15_settings"
]

# ==========================================
# ⚙️ VISUAL CONFIGURATION & EXCLUSIONS
# ==========================================

# 1. Skip strict pixel-by-pixel comparisons for these screens
EXCLUDED_SCREENS = [
    "14_dr_aria",          # Chatbot responses are dynamic and change constantly
    "11_digital_twin",     # Dynamic BioGears charts
]

# 2. Ignored Regions / Element Masking:
# Define bounding boxes [left, top, right, bottom] in pixels to black out (e.g. status bar, clock)
# This prevents dynamic data from triggering false visual mismatches.
IGNORED_REGIONS = {
    "global": [
        [0, 0, 1080, 80],   # Mask status bar at the top (time, battery, wifi icons)
    ],
    "03_dashboard": [
        [50, 320, 1030, 480],  # Mask the dynamic greeting and twin status card
    ],
}


def setup_directories():
    """Ensure all required test folders exist."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous run results and diffs
    for f in CURRENT_DIR.glob("*.png"):
        os.remove(f)
    for f in DIFF_DIR.glob("*.png"):
        os.remove(f)


def run_maestro_suite():
    """Execute Maestro test flow and assert screenshots are generated."""
    yaml_path = MAESTRO_DIR / "full_app_test.yaml"
    print(f"\n🚀 Running Maestro UI Macro test suite: {yaml_path.name}")
    
    # Run maestro command in the project root so it writes screenshots there
    result = subprocess.run(["maestro", "test", str(yaml_path)], cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print("\n❌ Maestro test run failed. Visual regression aborted.")
        sys.exit(result.returncode)


def collect_screenshots():
    """Move screenshots from run folder into current_run directory."""
    print("\n📦 Collecting captured screenshots...")
    found_any = False
    for name in SCREENSHOTS:
        src = PROJECT_ROOT / f"{name}.png"
        dest = CURRENT_DIR / f"{name}.png"
        if src.exists():
            shutil.move(str(src), str(dest))
            found_any = True
        else:
            print(f"   ⚠️ Warning: Screenshot '{name}.png' was not captured.")
            
    if not found_any:
        print("❌ Error: No screenshots were found in the workspace root.")
        sys.exit(1)


def mask_ignored_elements(img: Image.Image, screen_name: str) -> Image.Image:
    """Mask out specified dynamic areas with a solid black rectangle."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Apply global masks
    for box in IGNORED_REGIONS.get("global", []):
        draw.rectangle(box, fill=(0, 0, 0))
        
    # Apply screen-specific masks
    for box in IGNORED_REGIONS.get(screen_name, []):
        draw.rectangle(box, fill=(0, 0, 0))
        
    return img_copy


def check_overflow_and_bounds(img: Image.Image) -> bool:
    """
    Scans the left and right margins of the screenshot for overflow.
    If text/elements bleed directly to the very edge pixels of the screen width,
    it indicates clipping or out-of-bounds UI bugs.
    """
    w, h = img.size
    # Convert to grayscale to evaluate pixel intensity changes
    gray = img.convert("L")
    
    # We inspect the extreme left (x=0, x=1) and extreme right (x=w-1, x=w-2) columns
    # Ignoring the top 100px (header/status) and bottom 100px (nav bar/system keys)
    bleed_threshold_pixels = 0
    bg_color = gray.getpixel((0, h // 2))  # Estimate background color
    
    left_bleed = 0
    right_bleed = 0
    
    for y in range(120, h - 120):
        # Left margin check
        if abs(gray.getpixel((0, y)) - bg_color) > 30:
            left_bleed += 1
        # Right margin check
        if abs(gray.getpixel((w - 1, y)) - bg_color) > 30:
            right_bleed += 1
            
    # If more than 15 vertical pixels touch the boundary, flag layout bleed warning
    if left_bleed > 15 or right_bleed > 15:
        return False  # Failed fit test (bleed detected)
    return True  # Passed fit test


def compare_images(benchmark_path: Path, current_path: Path, diff_path: Path, screen_name: str) -> tuple[float, bool]:
    """Compare two images pixel-by-pixel with masking applied."""
    raw_b = Image.open(benchmark_path).convert("RGB")
    raw_c = Image.open(current_path).convert("RGB")
    
    # Dimension check
    if raw_b.size != raw_c.size:
        raw_c = raw_c.resize(raw_b.size)
        size_mismatch = True
    else:
        size_mismatch = False

    # Apply black masking to dynamic elements
    img_b = mask_ignored_elements(raw_b, screen_name)
    img_c = mask_ignored_elements(raw_c, screen_name)

    # Compute pixel diff
    diff = ImageChops.difference(img_b, img_c)
    
    diff_pixels = diff.getdata()
    mismatched_count = sum(1 for p in diff_pixels if sum(p) > 20)  # Threshold 20 to ignore compression artifacts
    total_pixels = img_b.size[0] * img_b.size[1]
    mismatch_pct = (mismatched_count / total_pixels) * 100
    
    if mismatched_count > 0 or size_mismatch:
        w, h = img_b.size
        composite = Image.new("RGB", (w * 3, h))
        composite.paste(img_b, (0, 0))
        
        diff_color = Image.new("RGB", img_b.size, (255, 0, 0))
        mask = diff.convert("L").point(lambda x: 255 if x > 20 else 0)
        highlighted_diff = Image.composite(diff_color, img_b, mask)
        
        composite.paste(highlighted_diff, (w, 0))
        composite.paste(img_c, (w * 2, 0))
        composite.save(diff_path)
        
    return mismatch_pct, size_mismatch


def perform_visual_regression():
    """Compare current screenshots with benchmarks. Handles exclusions and fit checks."""
    benchmarks = list(BENCHMARK_DIR.glob("*.png"))
    
    if not benchmarks:
        print("\n✨ No baseline screenshots found. Saving current build as benchmark baseline...")
        for img in CURRENT_DIR.glob("*.png"):
            shutil.copy2(str(img), str(BENCHMARK_DIR / img.name))
        print("\n✅ Benchmark established! Modify your code, then run again to test.")
        return True

    print("\n🔍 Evaluating UI consistency, exclusions, and bounds...")
    failures = 0
    passed = 0
    
    for name in SCREENSHOTS:
        b_file = BENCHMARK_DIR / f"{name}.png"
        c_file = CURRENT_DIR / f"{name}.png"
        d_file = DIFF_DIR / f"{name}_diff.png"
        
        if not c_file.exists():
            continue
            
        # 1. Overflow Out-of-bounds Check (Run on ALL pages)
        img_current = Image.open(c_file)
        fit_passed = check_overflow_and_bounds(img_current)
        
        # 2. Excluded Screen Check
        is_excluded = name in EXCLUDED_SCREENS
        
        if is_excluded:
            # If excluded, skip pixel regression, but check fitting bounds
            if not fit_passed:
                print(f"   ⚠️ WARNING: {name}.png [EXCLUDED from pixel diff] - Out-of-bounds overflow detected at margins!")
                failures += 1
            else:
                print(f"   ✅ SKIP: {name}.png [EXCLUDED from pixel diff] - Fitting bounds OK")
                passed += 1
            continue

        if not b_file.exists():
            shutil.copy2(str(c_file), str(b_file))
            passed += 1
            continue
            
        # 3. Pixel-level Comparison (Runs on non-excluded pages)
        mismatch_pct, size_mismatch = compare_images(b_file, c_file, d_file, name)
        
        THRESHOLD = 0.15 # 0.15% threshold allowed
        
        if mismatch_pct > THRESHOLD or size_mismatch or not fit_passed:
            status = "❌ FAIL"
            failures += 1
            err_msg = ""
            if mismatch_pct > THRESHOLD: err_msg += f" Mismatch {mismatch_pct:.2f}%."
            if size_mismatch: err_msg += " Dimension mismatch."
            if not fit_passed: err_msg += " Out-of-bounds edge bleeding."
            print(f"   {status}: {name}.png -{err_msg}")
        else:
            status = "✅ PASS"
            passed += 1
            print(f"   {status}: {name}.png - Diff: {mismatch_pct:.2f}% (Fit: OK)")
            
    print("\n" + "="*50)
    print(f"VISUAL TESTS SUMMARY:")
    print(f"   Passed: {passed}")
    print(f"   Failed / Warnings: {failures}")
    print("="*50)
    
    if failures > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    setup_directories()
    run_maestro_suite()
    collect_screenshots()
    perform_visual_regression()
