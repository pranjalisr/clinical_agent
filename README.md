# Discharge Summary Agent

> An agentic AI system that transforms messy, incomplete clinical source notes into safe, structured discharge summary drafts for clinician review.

---

## Quick Start (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Open .env and replace: DEEPSEEK_API_KEY=your_actual_key_here

# 3. Run
python main.py --patient patient_2
```

---

## Where to Add Your DeepSeek API Key

1. Open `.env` (copy from `.env.example` if it doesn't exist)
2. Find this line:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```
3. Replace `your_deepseek_api_key_here` with your actual key from [https://platform.deepseek.com](https://platform.deepseek.com)

Your `.env` should look like:
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
MAX_STEPS=20
MAX_RETRIES=3
```

---

## Usage

```bash
# Run on patient_2 (the provided patient)
python main.py --patient patient_2

# Run on all patients in data/patients/
python main.py

# Full demo: patient_2 + synthetic conflict patient
python main.py --demo

# With Part 2 learning loop (3 iterations)
python main.py --patient patient_2 --part2

# Full demo with learning loop
python main.py --demo --part2 --part2-iterations 5
```

---

## Output Files

After running, check the `output/` folder:

| File | Description |
|------|-------------|
| `patient_2_discharge_summary.txt` | Human-readable discharge summary draft |
| `patient_2_discharge_summary.json` | Machine-readable structured JSON |
| `patient_2_trace.json` | Full agent step trace (reasoning → action → result) |
| `part2_learning_report.txt` | Part 2 improvement curve and metrics |
| `part2_results.json` | Raw learning loop results |

---

## Project Structure

```
discharge_agent/
├── main.py                      # Entry point — run this
├── requirements.txt
├── .env.example                 # Template for API key
├── agents/
│   ├── discharge_agent.py       # Core agentic loop (Part 1)
│   └── learning_loop.py         # Simulated reviewer + learning (Part 2)
├── tools/
│   ├── pdf_ingestion.py         # PDF reading + OCR
│   └── clinical_tools.py        # Drug interactions, allergy checks, escalation
├── utils/
│   ├── models.py                # Pydantic data models for discharge summary
│   └── tracer.py                # Observability — step-by-step trace logger
└── data/
    └── patients/
        └── patient_2/
            └── patient_2.pdf    # Source clinical notes
```

---

## Agent Loop Design

The agent uses a **ReAct-style loop** (Reason → Act → Observe → Repeat) with DeepSeek as the backbone LLM and OpenAI-compatible function calling.

### Step Flow

```
START
  │
  ▼
[1] ingest_pdfs
  │   → OCR all PDFs in patient folder
  │   → Returns raw text per document
  │
  ▼
[2..9] extract_clinical_data (one call per extraction target)
  │   → demographics
  │   → diagnoses (with conflict detection)
  │   → vitals
  │   → medications_admission
  │   → medications_discharge
  │   → lab_results
  │   → hospital_course
  │   → allergies
  │   → follow_up
  │   → imaging
  │
  ▼
[10] check_drug_interactions
  │   → Checks all discharge medications
  │   → Flags any MAJOR/MODERATE interactions
  │
  ▼
[11] check_allergy_conflicts
  │   → Cross-references allergies vs medications
  │   → Flags CONTRAINDICATED combinations
  │
  ▼
[12] lookup_pending_results
  │   → Checks for outstanding lab results
  │
  ▼
[13] escalate_for_clinician_review (as needed)
  │   → Called whenever a safety concern is found
  │   → Creates formal escalation record
  │
  ▼
[14] compile_discharge_summary
  │   → Builds final structured DischargeSummary object
  │   → Runs medication reconciliation
  │   → Flags all MISSING fields
  │
  ▼
END — Saves .txt, .json, trace.json
```

### Hard Constraints

- **Step cap**: Hard limit of 20 steps (configurable via `MAX_STEPS` in `.env`). If reached, forces compilation with all available data and flags the truncation.
- **Retry logic**: Every tool call retried up to 3 times on failure. After max retries, failure is noted in the trace and the agent continues.
- **No fabrication**: The LLM extraction prompt explicitly forbids inventing values. Every missing field is marked `[[MISSING - CLINICIAN REVIEW REQUIRED]]`.

---

## No-Fabrication Guardrail

This is the most critical safety requirement. It is enforced in three layers:

**Layer 1 — System Prompt**
```
"If a clinical fact is NOT explicitly stated in the source documents,
you MUST mark it as [[MISSING - CLINICIAN REVIEW REQUIRED]].
You NEVER guess, infer, or generate plausible-sounding clinical values."
```

**Layer 2 — Extraction Prompt**
Each `extract_clinical_data` call is prefaced with:
```
"CRITICAL RULES:
- Only extract what is EXPLICITLY stated.
- Mark anything not found as MISSING or PENDING.
- Do NOT infer, guess, or generate plausible values."
```

**Layer 3 — Output Validation (models.py)**
The `DischargeSummary` Pydantic model uses sentinel values:
- `MISSING = "[[MISSING - CLINICIAN REVIEW REQUIRED]]"` as field defaults
- Any field that survives to the output with this value is explicitly flagged in the FLAGS section
- The output header clearly states `⚠ FOR CLINICIAN REVIEW ONLY ⚠`

---

## Handling Failures and Conflicts

### Tool Failures
- Retried up to `MAX_RETRIES` (default 3) times with backoff
- If all retries fail, the failure is logged in the trace and the agent continues
- The agent never treats a failed call as a successful one

### Conflicting Information
- The extraction prompt instructs the LLM to populate a `conflicts` field when documents disagree
- Conflicts are stored in `DischargeSummary.conflicts_detected`
- Each conflict is displayed in the FLAGS section with both values and both sources
- An escalation is raised via `escalate_for_clinician_review`
- **The agent never arbitrarily picks one conflicting value**

### Missing Data
- Required fields default to `MISSING` sentinel
- After compilation, all fields still holding `MISSING` are checked and flagged
- Pending results (e.g., urine culture) are stored in `pending_results` and shown prominently

### Medication Reconciliation
- Admission medications are compared against discharge medications
- New medications with no documented indication → flagged
- Admission medications not in discharge list → flagged as possible omission
- Drug interaction checker is always run before compilation

---

## Part 2: Learning from Doctor Edits

### Design

The learning mechanism uses **correction memory with pattern injection**:

1. **Reward Signal**: `reward = 1.0 - normalized_edit_distance(draft, edited)`. Computed at word level. Higher reward = fewer edits needed.

2. **Simulated Reviewer**: A separate LLM call with a fixed "doctor editing policy" that applies consistent, documented corrections (style, terminology, structure, completeness).

3. **Learning Mechanism**: 
   - Each (draft, edited) pair stored in `correction_memory.json`
   - After accumulating pairs, the LLM extracts recurring correction patterns
   - These patterns are injected into the extraction system prompt as few-shot guidance
   - Future drafts "pre-apply" common doctor corrections

4. **Evaluation**: Normalized edit distance + per-section accuracy, measured before/after across iterations.

### Limitations (Honest Discussion)

**Cold-start problem**: With one patient, patterns are anecdotal. Real deployment needs ≥50 patient pairs for generalizable patterns. Mitigation: seed with domain expert-written correction examples.

**Gaming risk**: An agent can lower edit distance by writing vaguer content (e.g., "see clinical notes" instead of specific values). This lowers edit distance without being more correct. Mitigation: reward must be a composite of (edit_distance × completeness_score). Completeness is checked by counting non-MISSING fields.

**Reviewer consistency**: The simulated doctor applies a fixed policy. Real clinicians disagree. Production systems need multi-annotator agreement (Fleiss κ > 0.7) before using edit distance as signal.

**Safety invariants must be preserved**: The no-fabrication and escalation rules are **hardcoded in the agent core** — they cannot be learned away. The learning loop only affects style and structure, never the safety layer.

---

## What I'd Do With More Time

1. **Multi-patient evaluation**: Run on 10+ synthetic patients, report aggregate metrics
2. **Structured output validation**: Add a post-generation validator that checks every field against source text (regex + semantic similarity)
3. **Confidence scores**: Have the LLM output 0-1 confidence per extracted field; low-confidence fields get flagged even if a value was found
4. **Section-aware OCR**: Use document layout analysis to segment PDFs by section before extraction
5. **Human-in-the-loop API**: Build a simple web UI where a clinician can confirm/reject each flag, closing the feedback loop properly
6. **DPO fine-tuning**: With enough (draft, edited) pairs, fine-tune the model directly rather than prompt-injecting patterns

---

## Dependencies

```
openai          # DeepSeek is OpenAI API-compatible
pdf2image       # Convert PDF pages to images
pytesseract     # OCR text from images
Pillow          # Image processing
python-dotenv   # Load .env file
rich            # Beautiful terminal output
pydantic        # Data validation for discharge summary model
```

System requirements: `tesseract`, `poppler-utils` (for `pdfinfo`, `pdftotext`).

On Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

On macOS:
```bash
brew install tesseract poppler
```

---

*All patient data in this project is synthetic. No real patient data was used.*
