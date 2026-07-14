---
name: prettier_markdown
description: Guidelines and instructions for creating high-fidelity, visually stunning, and highly professional Markdown documentation.
---

# Premium Markdown Styling Guidelines

Use this skill when creating or editing Markdown documentation (such as READMEs, upgrades reports, research logs, and walkthroughs). The goal is to provide a premium, state-of-the-art visual appearance that looks polished, scans easily, and reads like a professional design.

---

## 1. Typographical Hierarchy

- **Title (`#`):** Single H1 title per document. Use a bold, clear name.
- **Section Headers (`##`):** Use for major topics. Always prefix with a relevant emoji (e.g., `🚀 Overview`, `🛡️ Security`) to make sections distinct.
- **Minimal Emojis:** Keep emoji usage to a strict minimum (typically only in major section headers) to maintain a highly professional, clinical tone. Avoid scattering emojis inline within paragraphs or tables.
- **Sub-headers (`###`):** Use for sub-topics. Limit to three levels of heading nesting to prevent confusion.
- **Spacers (`---`):** Use horizontal lines to separate major thematic sections and avoid continuous page scroll fatigue.

---

## 2. Strategic GitHub Alerts

Use strategic alert blocks to break up text walls and draw focus to critical architectural caveats, details, or steps.

```markdown
> [!NOTE]
> For supplementary context, architecture tips, or helpful tips.

> [!IMPORTANT]
> For requirements, configurations, or parameters that are critical for execution.

> [!WARNING]
> For caveats, risks, or potential startup error codes.
```

---

## 3. Structured Data Tables

Avoid lists for tabular comparisons. Always organize feature comparisons, file status directories, or performance metrics into well-formatted Markdown tables.
- Bold the row keys.
- Keep columns aligned.
- Use explicit visual states (e.g., `[UPGRADED]`, `[SAFE]`, `[NEW]`).

---

## 4. Visual Architecture Diagrams (Mermaid)

When documenting logic paths, pipelines, or code architectures (like OCR pipelines or RAG workflows), **always include a Mermaid flow diagram**.
- Keep flow layouts logical (`graph TD` or `graph LR`).
- Style nodes using clear borders or text labels.
- Wrap complex names or paths in quotes to prevent syntax parsing issues.

---

## 5. Rich Code Formatting

- Always specify the language tag for syntax highlighting (e.g., `python`, `bash`, `javascript`, `diff`, `json`).
- Highlight modifications using `diff` blocks where comparative illustration is helpful.
- Reference actual files using clickable file links with line numbers if appropriate, formatted as `[filename](file:///path/to/file#L123-L135)`.
