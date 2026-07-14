# UHI Internship Contribution Repository

This repository hosts two primary project contributions developed during the UHI Internship for the Health Digital Twin platform:

1. **[Healthbot](Healthbot/)**: An offline-first clinical assistant backend and playground UI (featuring **Dr. Aria**). It runs local LLM generation (`Qwen3-30B-A3B.gguf`) with hybrid vector retrieval, prompt safety defenses, and robust document classifiers.
2. **[Macro](Macro/)**: A mobile UI automated quality assurance suite built using **Maestro** to validate onboarding flows, prescription logging, telemetry widgets, and chatbot interactions on the **VitalHealth** Expo application. *(Note: Only the profile onboarding flow is fully verified; all other flows are in testing phases).*

---

## Workspace Structure

The project has been structured into a clean monorepo format:

```
.
├── Healthbot/            # Backend server & local developer UI
│   ├── health_ai/        # Core LLM loaders, RAG engines, and safety routers
│   ├── chat.html         # Developer playground UI
│   ├── requirements.txt  # Python requirements and dependencies
│   ├── UPGRADES.md       # Chronological log of architectural upgrades
│   └── README.md         # Setup instructions and architecture flowcharts
│
├── Macro/                # Mobile app automated test macros
│   ├── AppMacro/         # Local configs and logging output landing area
│   ├── VitalHealthMacro/ # Maestro YAML flow files and timing scripts
│   └── README.md         # Macro execution instructions
│
├── .gitignore            # Monorepo global ignore rules
└── README.md             # This main index page
```

For setup guidelines, installation steps, and test suites, please refer directly to the documentation inside the subfolders:
- To run the chatbot API and playground, read the [Healthbot Documentation](Healthbot/README.md).
- To execute the mobile automation macros, read the [Macro Documentation](Macro/README.md).

---

## Disclaimer
This project is built on top of an existing baseline codebase. The author did not develop the application from the ground up. Instead, the author's work focuses on implementing the specific architectural enhancements, security upgrades, and dynamic optimizations detailed in [Healthbot/UPGRADES.md](Healthbot/UPGRADES.md), and fully re-designing the PDF document reader and OCR pipeline from scratch.
