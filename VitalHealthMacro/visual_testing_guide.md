# Visual Layout & Consistency Testing Guide

To confirm that buttons don't overlap, labels fit inside cards, and the UI layout is **100% consistent across builds**, you can use our automated **Visual Regression Tool** inside the `.maestro` folder.

---

## 🚀 Step 1: Run the Automated Visual Comparison Tool

We created a custom python tool `compare_ui_builds.py` that handles everything:
1. Executes `maestro test` using [full_app_test.yaml](file:///c:/Users/Zap/UHI_Internship/AppMacro/health-digital-twin/VitalHealth/.maestro/full_app_test.yaml) to click through every screen in the app.
2. Captures 15 specific screenshots (e.g. welcome screen, dashboard, settings, AI chatbot, medication vault).
3. If this is the **first run**, it establishes these screenshots as your **Baseline Benchmark** inside `.maestro/visual_regression/benchmarks/`.
4. On **subsequent runs** (after you change your code or build a new version), it compares the new screenshots pixel-by-pixel against the benchmarks.
5. If any pixel mismatches or layout shifts are detected:
   * It flags the failure.
   * It generates a composite **diff image** highlighting the changes in **bright red**.
   * Saves the diffs under `.maestro/visual_regression/diffs/`.

### How to Run:
Navigate to the `.maestro` folder and run the script:
```bash
python VitalHealth/.maestro/compare_ui_builds.py
```

---

## 📈 Visual Regression Workflow

```
[Current Build] ──► run compare_ui_builds.py ──► Saves benchmarks/
                                                       │
                                                       ▼
[Make UI Edits] ──► run compare_ui_builds.py ──► Compares new screenshots
                                                 against benchmarks/
                                                       │
                                         ┌─────────────┴─────────────┐
                                         ▼                           ▼
                                  No Layout Shifts            Layout Shifted!
                                  [✅ PASS (exit 0)]          [❌ FAIL (exit 1)]
                                                              View: diffs/ folder
```

---

## 🔍 Visual Difference Reports

If the layout changes, check `.maestro/visual_regression/diffs/`. Each report displays a side-by-side comparison:
1. **Left**: Benchmark baseline (known good layout).
2. **Middle**: Mismatches highlighted in **bright red** on top of the layout.
3. **Right**: The new build layout (showing the visual bug).

This allows you to catch even a single-pixel shift or overlapping text block instantly before launching the beta version.
