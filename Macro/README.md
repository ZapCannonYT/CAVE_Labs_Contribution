# VitalHealth Maestro QA Macros

This directory contains automated user-interaction testing macros written in the **Maestro** mobile UI automation YAML syntax. These scripts simulate clinical workflows and verify the stability of the **VitalHealth** React Native client application.

---

## Directory Structure

### AppMacro
This folder acts as a landing area for emulator logs and build configurations:
- `package-lock.json`
- `emulator_stderr.log`
- `emulator_stdout.log`

### VitalHealthMacro
Contains the core Maestro test flows, JavaScript assertions, and timing helpers:
- **`01_profile_onboarding.yaml`**: Automates patient account creation, gender/age configuration, and onboarding settings.
- **`02_dashboard_telemetry.yaml`**: Validates layout rendering of dashboard widgets (Steps, Heart Rate, SpO2, Calories).
- **`03_medication_vault.yaml`**: Tests the flow of adding a prescription, configuring frequencies, and verifying vault listings.
- **`04_symptom_logging.yaml`**: Simulates logging clinical symptoms (e.g. pain level, chest tight) and saving clinical inputs.
- **`05_chatbot_aria.yaml`**: Emulates an interactive chat session with the Dr. Aria AI medical assistant.
- **`sub_flows/`**: Directory containing reusable flow segments (e.g., `swipe_hour.yaml`, `02_personal_info.yaml`).
- **`sleep1s.js` / `sleep500ms.js` / `checkTime.js`**: JavaScript scripting blocks to handle timing constraints and dynamic checks.

> [!NOTE]
> **Testing Phase Status:** The `01_profile_onboarding.yaml` flow is fully verified and stable. All other automation scripts (from dashboard telemetry to chatbot interaction) are currently in testing phases.

---

## How to Run the Macros

1. **Prerequisites:**
   - Install the Maestro CLI tools locally (refer to the official [Maestro Documentation](https://maestro.mobile.dev/)).
   - Ensure an Android Emulator or iOS Simulator is running with the **VitalHealth** application installed.

2. **Execute a Flow:**
   Navigate to the `VitalHealthMacro` directory and run your target flow:
   ```bash
   maestro test 01_profile_onboarding.yaml
   ```

3. **Run All Tests:**
   Execute the full suite using:
   ```bash
   maestro test full_app_test.yaml
   ```
