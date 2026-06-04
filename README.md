# Discharge Summary Agent

> An agentic AI system that transforms messy, real-world clinical source notes into safe, structured discharge summary drafts. Built for clinician review, never for autonomous use.

---

## The Problem This Solves

Clinical discharge summaries are high-stakes documents. In real hospitals they are often assembled manually from scattered notes, handwritten consultation sheets, scanned lab reports, and poorly structured EHR entries. Errors: missed medications, wrong diagnoses, undocumented allergies cause patient harm.

This system ingests those messy source documents, extracts structured clinical information using an agentic AI loop, runs automated safety checks (drug interactions, allergy conflicts, medication reconciliation), and produces a structured draft with every uncertain field explicitly flagged for a clinician to verify.

**The core design principle: the system never guesses. Every fact it cannot source is marked `[[MISSING - CLINICIAN REVIEW REQUIRED]]`.**

---

## Quick Start

```bash
# 1. Install system dependencies (macOS)
brew install tesseract poppler

# On Ubuntu/Debian instead:
# sudo apt-get install tesseract-ocr poppler-utils

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Add your DeepSeek API key
cp .env
# Open .env and set: DEEPSEEK_API_KEY=your_actual_key_here

# 4. Run
python main.py --demo --part2
```

---

## Adding Your DeepSeek API Key

1. Copy the template: `cp .env`
2. Open `.env` in any text editor
3. Replace `your_deepseek_api_key_here` with your real key

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
MAX_STEPS=20
MAX_RETRIES=3
```

Get your key at: https://platform.deepseek.com

The `.env` file is in `.gitignore` — your key will never be committed.

---

## Usage

```bash
# Run on the provided patient (patient_2)
python main.py --patient patient_2

# Run on all patients in data/patients/
python main.py

# Full demo: patient_2 + synthetic conflict patient
python main.py --demo

# Run with Part 2 learning loop (3 iterations)
python main.py --patient patient_2 --part2

# Full demo with learning loop
python main.py --demo --part2

# Custom number of learning iterations
python main.py --demo --part2 --part2-iterations 5
```

---

## Output Files

All outputs are saved to `output/` after each run.

| File | What it contains |
|------|-----------------|
| `patient_2_discharge_summary.txt` | Human-readable discharge summary draft |
| `patient_2_discharge_summary.json` | Machine-readable structured JSON (all fields) |
| `patient_2_trace.json` | Full agent trace — every step, reasoning, action, result |
| `patient_demo_discharge_summary.txt` | Demo conflict patient summary |
| `patient_demo_discharge_summary.json` | Demo conflict patient JSON |
| `part2_learning_report.txt` | Learning loop improvement report |
| `part2_results.json` | Per-iteration reward and section accuracy scores |
| `correction_memory.json` | Accumulated doctor correction patterns |

---

## Project Structure

```
discharge_agent/
│
├── main.py                        # Entry point — run this
├── requirements.txt               # Python dependencies
├── .env                           # API key template (copy to .env)
├── .gitignore
│
├── agents/
│   ├── discharge_agent.py         # Core agentic loop (Part 1)
│   └── learning_loop.py           # Doctor review simulation + learning (Part 2)
│
├── tools/
│   ├── pdf_ingestion.py           # PDF reading, OCR, plain text ingestion
│   └── clinical_tools.py          # Drug interactions, allergy checks, escalation, pending results
│
├── utils/
│   ├── models.py                  # Pydantic schema for the discharge summary
│   └── tracer.py                  # Step-by-step observability logger
│
└── data/
    └── patients/
        └── patient_2/
            └── patient_2.pdf      # Source clinical notes (71-page scanned PDF)
```

---

## How the Agent Works

The agent uses a **ReAct-style loop** — Reason, Act, Observe, Repeat — powered by DeepSeek via the OpenAI-compatible API with function calling.

### The Loop

At every step the agent:
1. Reasons about what it still needs
2. Chooses a tool to call
3. Receives the result
4. Decides what to do next

It is not a fixed pipeline. The agent decides the order and number of tool calls.

### Tool Set

| Tool | What it does |
|------|-------------|
| `ingest_pdfs` | Reads all PDFs in the patient folder. Auto-detects scanned documents and runs Tesseract OCR. Also handles plain text files. |
| `extract_clinical_data` | Calls DeepSeek to extract one clinical domain at a time (demographics, diagnoses, vitals, medications, labs, imaging, etc.) from the OCR text. |
| `check_drug_interactions` | Checks all discharge medications for drug-drug interactions. |
| `check_allergy_conflicts` | Cross-references patient allergies against prescribed medications. |
| `lookup_pending_results` | Checks for lab or culture results still awaiting return. |
| `escalate_for_clinician_review` | Creates a formal escalation record for any safety concern, conflict, or missing critical data. |
| `compile_discharge_summary` | Builds the final `DischargeSummary` object, runs medication reconciliation, flags all missing fields. |

### Typical Step Sequence

```
Step 1  →  ingest_pdfs
Step 2  →  extract_clinical_data (demographics)
Step 2  →  extract_clinical_data (diagnoses)
Step 2  →  extract_clinical_data (allergies)
Step 2  →  extract_clinical_data (vitals)
Step 3  →  extract_clinical_data (medications_admission)
Step 3  →  extract_clinical_data (medications_discharge)
Step 3  →  extract_clinical_data (lab_results)
Step 3  →  extract_clinical_data (hospital_course)
Step 3  →  extract_clinical_data (follow_up)
Step 3  →  extract_clinical_data (imaging)
Step 4  →  check_drug_interactions
Step 4  →  check_allergy_conflicts
Step 5  →  lookup_pending_results
Step 6  →  escalate_for_clinician_review  (for each issue found)
Step 7  →  compile_discharge_summary
```

The agent may reorder or repeat steps based on what it finds.

### Hard Safety Constraints

**Step cap.** The loop is hard-limited to 20 steps (`MAX_STEPS` in `.env`). If the limit is reached before compilation, the agent forces compilation with whatever data it has and adds a flag noting the truncation.

**Retry logic.** Every tool call is retried up to 3 times (`MAX_RETRIES`) on failure with backoff. After all retries fail, the failure is recorded in the trace and the agent continues — it never silently drops a failed check.

**No fabrication.** This constraint is enforced at three layers — see below.

---

## The No-Fabrication Guardrail

This is the most important safety property of the system.

### Layer 1 — System Prompt (agents/discharge_agent.py)

```
## YOUR CORE MANDATE: NEVER FABRICATE CLINICAL FACTS

You have ONE absolute rule that overrides everything else:
- If a clinical fact is NOT explicitly stated in the source documents,
  you MUST mark it as [[MISSING - CLINICIAN REVIEW REQUIRED]]
- You NEVER guess, infer, or generate plausible-sounding clinical values.
- Every output is a DRAFT for clinician review, never a finalized document.
```

### Layer 2 — Per-Extraction Prompt (agents/discharge_agent.py)

Every `extract_clinical_data` call includes:

```
CRITICAL RULES:
- Only extract what is EXPLICITLY stated in the text below.
- Mark anything not found as MISSING or PENDING.
- Do NOT infer, guess, or generate plausible values.
- You MUST return a JSON object even if all values are MISSING.
- NEVER return an empty string.
```

### Layer 3 — Pydantic Model Defaults (utils/models.py)

```python
MISSING = "[[MISSING - CLINICIAN REVIEW REQUIRED]]"

class DischargeSummary(BaseModel):
    patient_name: str = MISSING
    date_of_admission: str = MISSING
    principal_diagnosis: str = MISSING
    hospital_course: str = MISSING
    discharge_condition: str = MISSING
    ...
```

Every critical field defaults to the `MISSING` sentinel. Fields that remain `MISSING` after compilation are automatically added to the FLAGS section. The output document always carries the header:

```
⚠  FOR CLINICIAN REVIEW ONLY ⚠
```

---

## Safety Checks

### Drug Interaction Checking

After discharge medications are identified, the agent automatically calls `check_drug_interactions`. Any MAJOR or MODERATE interaction is stored in the summary and displayed in the FLAGS section.

Example interactions checked: warfarin + aspirin, metformin + contrast, ofloxacin + metformin, pantoprazole + clopidogrel.

### Allergy Conflict Checking

The agent cross-references known patient allergies against all discharge medications. Example caught by the demo patient:

```
🔴 ALLERGY CONFLICT
Patient has documented NSAID allergy.
Aspirin 75mg prescribed as NEW discharge medication.
→ CONTRAINDICATED — clinician review required.
```

### Medication Reconciliation

After compilation, every admission medication is compared against the discharge medication list:

- Medications added at discharge with no documented reason → flagged
- Admission medications absent from discharge list → flagged as possible omission
- Changed doses → flagged

### Pending Results

The agent calls `lookup_pending_results` before compiling. Any outstanding lab or culture result is surfaced in the PENDING RESULTS section with a note that the patient should follow up.

---

## Conflict Detection

When source documents contradict each other, the agent detects and escalates the conflict rather than silently picking one value.

Example from the demo patient:

```
🚨 ESCALATION FLAGGED
CONFLICTING PRINCIPAL DIAGNOSIS
  Admission Note:    Unstable Angina
  Progress Note D3:  Musculoskeletal Chest Pain
  → [[CONFLICT - CLINICIAN REVIEW REQUIRED]]
```

Both values are preserved in the output. A clinician must resolve which is correct before the document is finalized.

---

## Observability

Every agent run produces a `trace.json` file capturing the full step-by-step log:

```json
{
  "step": 6,
  "timestamp": "2026-06-03T17:14:59",
  "duration_ms": 1842.3,
  "reasoning": "Patient has documented NSAID allergy but Aspirin is in the discharge list. Escalating.",
  "action": "escalate_for_clinician_review",
  "inputs": {
    "reason": "Aspirin prescribed despite documented NSAID allergy",
    "severity": "CRITICAL"
  },
  "result": {
    "success": true,
    "escalation_id": "ESC-1780487130"
  },
  "next_decision": "Continue extracting data"
}
```

This trace enables full audit of every decision the agent made, in order, with timing.

---

## Part 2: Learning from Doctor Edits

### Overview

After generating a discharge summary, the system can run a simulated improvement loop that measures how much clinician editing the draft requires, extracts recurring correction patterns, and injects those patterns into future runs.

### How It Works

**Step 1 — Simulated doctor review.**
A separate LLM call applies a fixed, documented editing policy to the draft. The policy specifies consistent rules: rewrite passive phrasing, fix clinical terminology, make follow-up instructions specific, ensure medication list completeness, and so on.

**Step 2 — Reward signal.**
```
reward = 1.0 - normalized_edit_distance(draft, edited)
```
Computed at word level. `reward = 1.0` means no edits were needed. `reward = 0.0` means the draft was completely rewritten.

**Step 3 — Correction memory.**
Each `(draft, edited)` pair is stored in `correction_memory.json`. After accumulating pairs, the LLM is asked to identify recurring correction patterns across the pairs.

**Step 4 — Pattern injection.**
Identified patterns are prepended to the extraction prompts for the next run, so the agent pre-applies common doctor corrections before generating the draft.

**Step 5 — Measurement.**
Edit distance and per-section accuracy are recorded for each iteration, producing an improvement curve.

### Actual Results (3 iterations on demo patient)

```
BEFORE (Iteration 1):  Edit distance: 0.783  |  Reward: 0.217
AFTER  (Iteration 3):  Edit distance: 0.666  |  Reward: 0.334

Edit distance Δ: +0.117  ✓ IMPROVED
Reward Δ:        +0.117

SECTION ACCURACY (Final Iteration):
  PATIENT INFORMATION  : 1.000  ████████████████████
  DIAGNOSES            : 0.545  ███████████
  HOSPITAL COURSE      : 0.262  █████
  DISCHARGE MEDICATIONS: 0.500  ██████████
  PENDING RESULTS      : 1.000  ████████████████████
```

### Limitations and Honest Discussion

**Cold-start.** With one patient, learned patterns are case-specific. Generalizable patterns require at least 50 patient pairs. Mitigation for production: seed the memory with expert-written correction examples.

**Gaming risk.** An agent could reduce edit distance by producing vaguer content — fewer specific words means fewer edits. The correct reward must combine edit distance with a completeness score (count of non-MISSING fields). This mitigation is documented but not yet implemented.

**Reviewer consistency.** The simulated doctor applies a fixed deterministic policy. Real clinicians disagree. A production system needs inter-annotator agreement (Fleiss κ > 0.7) before edit distance is meaningful as a training signal.

**Safety invariants are hardcoded.** The no-fabrication rule, the escalation requirement, and the MISSING sentinels are hardcoded in the agent core (`discharge_agent.py`). They are not part of the learned prompt. The learning loop only affects style, structure, and terminology — never the safety layer.

---

## What Would Be Added With More Time

**Structured output validation.** A post-generation step that checks every extracted value against the source text using regex and semantic similarity, rejecting values that cannot be grounded in the document.

**Confidence scores.** The extraction LLM outputs a 0–1 confidence score per field alongside the value. Fields below a threshold (e.g., 0.7) are flagged even when a value was found.

**Section-aware OCR.** Use document layout analysis (e.g., `layoutparser`, `pdfplumber`) to segment PDFs by visual section before extraction, so the extraction prompt receives only the relevant section rather than the full document.

**Multi-patient evaluation.** Run on 20+ synthetic patients and report aggregate precision/recall per clinical field across the corpus.

**Human-in-the-loop API.** A minimal web interface where a clinician confirms or rejects each flag, properly closing the feedback loop with real signal rather than simulated edits.

**DPO fine-tuning.** With sufficient `(draft, clinician-edited)` pairs, fine-tune the backbone model directly using Direct Preference Optimization rather than prompt injection.

---

## Dependencies

### Python packages

```
openai>=1.0.0        # DeepSeek uses the OpenAI-compatible API
pdf2image>=1.16.0    # Convert scanned PDF pages to images for OCR
pytesseract>=0.3.10  # OCR engine wrapper
Pillow>=10.0.0       # Image processing
python-dotenv>=1.0.0 # Load .env file
rich>=13.0.0         # Terminal output formatting
pydantic>=2.0.0      # Data validation and schema enforcement
```

### System packages

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu / Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

`tesseract` handles OCR on scanned PDFs. `poppler` provides `pdftotext` and `pdfinfo` used as the first-pass native text extraction attempt before falling back to OCR.

---

## Environment Variables

All configuration lives in `.env`. Copy `.env.example` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | — | **Required.** Your DeepSeek API key. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API endpoint. Do not change unless self-hosting. |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name. |
| `MAX_STEPS` | `20` | Hard cap on agent loop iterations. |
| `MAX_RETRIES` | `3` | Retries per tool call on failure. |

---

## Important Notes

This system produces **draft documents only**. All output files carry the header:

```
⚠  DRAFT — FOR CLINICIAN REVIEW ONLY ⚠
This document was auto-generated and must be reviewed and signed
by a qualified clinician before use in any clinical context.
```

The system is not a medical device. It is a productivity tool for trained clinical staff. No output should be used for patient care decisions without clinician verification.

The patient PDF included in this repository (`data/patients/patient_2/patient_2.pdf`) is a de-identified sample provided as part of the task specification. All synthetic patient data used in the demo is fictional.
