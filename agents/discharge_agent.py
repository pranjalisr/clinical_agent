"""
Discharge Summary Agent
=======================
A real agentic loop that:
1. Plans what to read and what tools to call
2. Ingests clinical PDFs (OCR if needed)
3. Extracts structured clinical information
4. Runs drug-interaction and allergy checks
5. Flags EVERYTHING it cannot source — never fabricates
6. Enforces a hard step cap
7. Emits a readable trace at every step
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

from tools.pdf_ingestion import ingest_patient_folder
from tools.clinical_tools import (
    check_drug_interactions,
    check_allergy_conflicts,
    escalate_for_clinician_review,
    lookup_pending_results,
)
from utils.tracer import AgentTracer
from utils.models import DischargeSummary, Medication, LabResult, Conflict, MISSING, PENDING

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ── Tool Definitions (OpenAI function-calling format) ─────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ingest_pdfs",
            "description": "Read and OCR all PDF files in the patient folder. Returns raw extracted text per document. Call this FIRST.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_folder": {"type": "string", "description": "Path to the patient's folder containing PDFs"}
                },
                "required": ["patient_folder"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_clinical_data",
            "description": "Parse the raw OCR text to extract structured clinical information: demographics, diagnoses, medications, vitals, lab results, hospital course. Only extract what is EXPLICITLY present — mark everything else as MISSING.",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string", "description": "DO NOT PASS THIS. Leave empty. The system retrieves text automatically from ingested documents."},
                    "extraction_target": {
                        "type": "string",
                        "enum": [
                            "demographics",
                            "diagnoses",
                            "vitals",
                            "medications_admission",
                            "medications_discharge",
                            "lab_results",
                            "hospital_course",
                            "allergies",
                            "follow_up",
                            "imaging",
                        ],
                        "description": "Which piece of clinical data to extract",
                    },
                },
                "required": ["extraction_target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_drug_interactions",
            "description": "Check for drug-drug interactions among a list of medications. Call when discharge medications are known.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of medication names to check",
                    }
                },
                "required": ["medications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_allergy_conflicts",
            "description": "Check if any medications conflict with known patient allergies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "allergies": {"type": "array", "items": {"type": "string"}, "description": "Known patient allergies"},
                    "medications": {"type": "array", "items": {"type": "string"}, "description": "Medications to check"},
                },
                "required": ["allergies", "medications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_pending_results",
            "description": "Check the lab system for any results that are still pending/awaited.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"}
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_for_clinician_review",
            "description": "Formally flag a safety concern, conflict, or missing critical data for mandatory clinician review. Use this whenever you find contradictions, dangerous drug interactions, allergy conflicts, or cannot source a critical clinical fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Clear description of the concern requiring review"},
                    "details": {"type": "object", "description": "Structured details about the concern"},
                    "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                },
                "required": ["reason", "details", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compile_discharge_summary",
            "description": "Compile all gathered information into the final structured discharge summary. Call this as the last step once all data has been extracted and checked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_data": {
                        "type": "object",
                        "description": "All discharge summary fields as a JSON object",
                    }
                },
                "required": ["summary_data"],
            },
        },
    },
]

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical AI assistant that produces DRAFT discharge summaries from raw patient notes.

## YOUR CORE MANDATE: NEVER FABRICATE CLINICAL FACTS

You have ONE absolute rule that overrides everything else:
- If a clinical fact (diagnosis, medication, lab value, date, procedure) is NOT explicitly stated in the source documents, you MUST mark it as [[MISSING - CLINICIAN REVIEW REQUIRED]] or [[PENDING - AWAITING RESULT]].
- You NEVER guess, infer, or generate plausible-sounding clinical values.
- Every output is a DRAFT for clinician review, never a finalized document.

## YOUR AGENTIC LOOP

You operate in a loop. At each step you:
1. Think about what information you still need
2. Choose the best tool to call
3. Process the result
4. Decide what to do next

## REQUIRED FIELDS FOR DISCHARGE SUMMARY
- Patient demographics (name, age, gender, MRN, blood group)
- Admission & discharge dates
- Principal diagnosis + secondary diagnoses
- Allergies
- Presenting complaints & history
- Vitals on admission
- Hospital course narrative
- Key investigations & lab results
- Imaging results
- Procedures performed
- Admission medications
- Discharge medications with changes clearly noted
- Discharge condition
- Discharge instructions
- Follow-up instructions
- Pending results

## HOW TO HANDLE PROBLEMS
- Missing data → mark as [[MISSING - CLINICIAN REVIEW REQUIRED]]
- Pending results → mark as [[PENDING - AWAITING RESULT]]
- Conflicting information → mark as [[CONFLICT - CLINICIAN REVIEW REQUIRED]] and escalate
- Drug interactions → escalate AND include in summary
- Medication change with no documented reason → flag for reconciliation
- Tool failure → retry once, then note failure and continue

## WORKFLOW ORDER
1. ingest_pdfs → get raw text
2. extract_clinical_data (multiple calls, one target at a time) → structured data
3. check_drug_interactions → safety check
4. check_allergy_conflicts → safety check  
5. lookup_pending_results → pending labs
6. escalate_for_clinician_review → for any issues found
7. compile_discharge_summary → final output

Be methodical. Be safe. Flag everything uncertain."""


class DischargeAgent:
    def __init__(self, patient_folder: str, patient_id: str, output_dir: str = "output"):
        self.patient_folder = patient_folder
        self.patient_id = patient_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracer = AgentTracer(patient_id, str(self.output_dir))
        self.summary = DischargeSummary(patient_id=patient_id)

        # State accumulated across steps
        self.raw_texts: dict[str, str] = {}
        self.extracted_data: dict[str, Any] = {}
        self.escalations_made: list[dict] = []

        # LLM client
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
            raise ValueError(
                "\n\n❌ No DeepSeek API key found!\n"
                "Please create a .env file with: DEEPSEEK_API_KEY=your_actual_key\n"
                "See .env.example for reference.\n"
            )

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── Tool Dispatch ─────────────────────────────────────────────────────────

    def _dispatch_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Execute a tool call with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                if tool_name == "ingest_pdfs":
                    return self._tool_ingest_pdfs(tool_args)
                elif tool_name == "extract_clinical_data":
                    return self._tool_extract_clinical_data(tool_args)
                elif tool_name == "check_drug_interactions":
                    result = check_drug_interactions(tool_args["medications"])
                    if not result["success"] and attempt < MAX_RETRIES - 1:
                        self.tracer.log_warning(f"Drug interaction tool failed (attempt {attempt+1}), retrying...")
                        time.sleep(0.5)
                        continue
                    return result
                elif tool_name == "check_allergy_conflicts":
                    result = check_allergy_conflicts(tool_args["allergies"], tool_args["medications"])
                    if not result["success"] and attempt < MAX_RETRIES - 1:
                        time.sleep(0.5)
                        continue
                    return result
                elif tool_name == "lookup_pending_results":
                    result = lookup_pending_results(tool_args["patient_id"])
                    if not result["success"] and attempt < MAX_RETRIES - 1:
                        time.sleep(0.5)
                        continue
                    return result
                elif tool_name == "escalate_for_clinician_review":
                    result = escalate_for_clinician_review(tool_args["reason"], tool_args.get("details", {}))
                    self.escalations_made.append(result)
                    self.tracer.log_escalation(tool_args["reason"], tool_args.get("details", {}))
                    self.summary.add_flag(
                        category="ESCALATION",
                        message=tool_args["reason"],
                        severity=tool_args.get("severity", "HIGH"),
                    )
                    return result
                elif tool_name == "compile_discharge_summary":
                    return self._tool_compile_summary(tool_args)
                else:
                    return {"error": f"Unknown tool: {tool_name}"}
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    self.tracer.log_warning(f"Tool {tool_name} raised exception: {e}. Retrying...")
                    time.sleep(0.5)
                else:
                    return {"error": f"Tool {tool_name} failed after {MAX_RETRIES} attempts: {str(e)}"}
        return {"error": "Max retries exceeded"}

    def _tool_ingest_pdfs(self, args: dict) -> dict:
        folder = args.get("patient_folder", self.patient_folder)
        results = ingest_patient_folder(folder)
        if "error" in results:
            return results

        combined_text = ""
        summary_parts = []
        for fname, res in results.items():
            if res["success"]:
                text = res["text"]
                self.raw_texts[fname] = text
                combined_text += f"\n\n===== DOCUMENT: {fname} =====\n{text}"
                summary_parts.append(f"{fname}: {res['page_count']} pages, method={res['method']}, chars={len(text)}")
            else:
                summary_parts.append(f"{fname}: FAILED - {res.get('error')}")
                self.tracer.log_warning(f"Failed to ingest {fname}: {res.get('error')}")

        self.extracted_data["combined_raw_text"] = combined_text
        return {
            "success": True,
            "documents_processed": len(results),
            "summary": summary_parts,
            "total_chars": len(combined_text),
        }

    def _tool_extract_clinical_data(self, args: dict) -> dict:
        """Use LLM to extract a specific piece of clinical data from the raw text."""
        # Always use internally stored OCR text. Ignore whatever the LLM passes
        # as raw_text — it frequently passes a file path or folder path instead.
        raw_text = self.extracted_data.get("combined_raw_text", "")
        target = args["extraction_target"]

        if not raw_text:
            return {"error": "No raw text available. Call ingest_pdfs first."}

        # Truncate text to avoid token limits
        # Take first 6000 chars (admission/diagnosis info) + last 6000 chars (discharge/meds)
        if len(raw_text) > 12000:
            text_snippet = raw_text[:6000] + "\n\n...[middle truncated]...\n\n" + raw_text[-6000:]
        else:
            text_snippet = raw_text

        if not text_snippet.strip():
            return {"error": "Raw text is empty after truncation. Ingestion may have failed.", "target": target}

        extraction_prompts = {
            "demographics": "Extract: patient name, age, gender, MRN/patient ID, blood group. Return JSON.",
            "diagnoses": "Extract: principal diagnosis, all secondary diagnoses. Note any conflicts between documents. Return JSON with 'principal', 'secondary' (list), 'conflicts' (list).",
            "vitals": "Extract: vitals on admission (BP, PR, RR, SpO2, temperature, blood sugar). Return JSON.",
            "medications_admission": "Extract: all medications the patient was on BEFORE/AT ADMISSION (pre-existing medications, home medications). Return JSON list with name, dose, frequency.",
            "medications_discharge": "Extract: all medications prescribed AT DISCHARGE. Return JSON list with name, dose, frequency, duration. Note which are new vs continued.",
            "lab_results": "Extract: all lab/blood test results with values, units, and reference ranges. Note ABNORMAL results. Return JSON list.",
            "hospital_course": "Write a concise hospital course narrative summarizing: why admitted, what treatment was given, how patient responded, any complications, and reason for discharge. Use ONLY facts from the documents.",
            "allergies": "Extract: all known drug or food allergies. If none documented, state that explicitly. Return JSON list.",
            "follow_up": "Extract: follow-up instructions, OPD appointments, wound care, diet, activity restrictions. Return JSON list.",
            "imaging": "Extract: all imaging/radiology/USG/ECG results and their findings. Return JSON list.",
        }

        prompt = f"""From these clinical notes, {extraction_prompts[target]}

CRITICAL RULES:
- Only extract what is EXPLICITLY stated in the text below.
- Mark anything not found as MISSING or PENDING.
- Do NOT infer, guess, or generate plausible values.
- If there are conflicts between documents, note them.
- You MUST return a JSON object or array even if all values are MISSING.
- NEVER return an empty string. If nothing found, return {{"found": false, "note": "not present in source documents"}}

CLINICAL NOTES (extracted via OCR):
---BEGIN TEXT---
{text_snippet}
---END TEXT---

Respond with clean JSON only, no markdown fences. Never return empty string."""
        try:
            resp = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You extract structured data from clinical notes. Return clean JSON only. Never fabricate clinical values."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            content = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:])
                if content.endswith("```"):
                    content = content[:-3]
            try:
                parsed = json.loads(content)
                self.extracted_data[target] = parsed
                return {"success": True, "target": target, "data": parsed}
            except json.JSONDecodeError:
                # Return raw string if not valid JSON
                self.extracted_data[target] = content
                return {"success": True, "target": target, "data": content, "note": "Non-JSON response"}
        except Exception as e:
            return {"error": f"LLM extraction failed for {target}: {str(e)}"}

    def _tool_compile_summary(self, args: dict) -> dict:
        """Build the final DischargeSummary from all extracted data."""
        data = args.get("summary_data", {})
        ed = self.extracted_data

        # Demographics
        demog = ed.get("demographics", {})
        if isinstance(demog, dict):
            self.summary.patient_name = demog.get("name") or demog.get("patient_name") or MISSING
            self.summary.age = demog.get("age") or demog.get("Age")
            self.summary.gender = demog.get("gender") or demog.get("Gender")
            self.summary.mrn = demog.get("mrn") or demog.get("MRN") or demog.get("patient_id")
            self.summary.blood_group = demog.get("blood_group") or demog.get("Blood Group")

        # Diagnoses
        diag = ed.get("diagnoses", {})
        if isinstance(diag, dict):
            self.summary.principal_diagnosis = diag.get("principal") or diag.get("primary") or MISSING
            sec = diag.get("secondary", [])
            if isinstance(sec, list):
                self.summary.secondary_diagnoses = [str(s) for s in sec if s]
            elif isinstance(sec, str) and sec:
                self.summary.secondary_diagnoses = [sec]
            # Check for diagnosis conflicts
            conflicts = diag.get("conflicts", [])
            if conflicts:
                for c in conflicts:
                    self.summary.add_flag("CONFLICT", f"Diagnosis conflict detected: {c}", "HIGH")

        # Vitals
        vitals = ed.get("vitals", {})
        if isinstance(vitals, dict):
            self.summary.vitals_on_admission = vitals

        # Allergies
        allergies = ed.get("allergies", [])
        if isinstance(allergies, list):
            # Filter out any dicts that slipped into the list
            self.summary.allergies = [str(a) for a in allergies if a and not isinstance(a, dict)]
        elif isinstance(allergies, dict):
            allergy_list = allergies.get("allergies", allergies.get("list", allergies.get("allergen", [])))
            if isinstance(allergy_list, list):
                self.summary.allergies = [str(a) for a in allergy_list if a]
            elif isinstance(allergy_list, str):
                self.summary.allergies = [allergy_list]
            note = allergies.get("note") or allergies.get("Note") or allergies.get("source")
            if note:
                self.summary.allergy_note = str(note)
        elif isinstance(allergies, str) and allergies:
            self.summary.allergies = [allergies]

        # Medications
        adm_meds = ed.get("medications_admission", [])
        if isinstance(adm_meds, list):
            self.summary.admission_medications = [
                Medication(
                    name=m.get("name", m) if isinstance(m, dict) else str(m),
                    dose=m.get("dose") if isinstance(m, dict) else None,
                    frequency=m.get("frequency") if isinstance(m, dict) else None,
                )
                for m in adm_meds if m
            ]

        disc_meds = ed.get("medications_discharge", [])
        if isinstance(disc_meds, list):
            adm_names = {m.name.lower() for m in self.summary.admission_medications}
            processed_meds = []
            for m in disc_meds:
                if isinstance(m, dict):
                    med = Medication(
                        name=m.get("name", ""),
                        dose=m.get("dose"),
                        frequency=m.get("frequency"),
                        duration=m.get("duration"),
                    )
                    med_lower = med.name.lower()
                    # Medication reconciliation
                    if med_lower and med_lower not in adm_names:
                        med.change_from_admission = "NEW"
                    elif m.get("change") or m.get("status") == "stopped":
                        med.change_from_admission = "CHANGED"
                    else:
                        med.change_from_admission = "CONTINUED"

                    # Flag if new with no documented reason
                    if med.change_from_admission == "NEW":
                        reason = m.get("reason") or m.get("indication")
                        if not reason:
                            med.flag = "New medication added at discharge — no documented indication in notes. Reconciliation required."
                            self.summary.add_flag(
                                "MEDICATION",
                                f"New discharge medication '{med.name}' has no documented indication/reason in source notes.",
                                "MEDIUM",
                            )
                    processed_meds.append(med)
                else:
                    processed_meds.append(Medication(name=str(m)))
            self.summary.discharge_medications = processed_meds

        # Stopped medications — any admission med not in discharge
        disc_names = {m.name.lower() for m in self.summary.discharge_medications}
        for adm_med in self.summary.admission_medications:
            if adm_med.name.lower() not in disc_names:
                self.summary.medication_reconciliation_notes.append(
                    f"Admission medication '{adm_med.name}' NOT found in discharge list. "
                    "Confirm if intentionally stopped, or if this is an omission error."
                )
                self.summary.add_flag(
                    "MEDICATION",
                    f"Admission medication '{adm_med.name}' absent from discharge medications — reason undocumented.",
                    "MEDIUM",
                )

        # Labs
        labs = ed.get("lab_results", [])
        if isinstance(labs, list):
            safe_labs = []
            for l in labs:
                if not l:
                    continue
                try:
                    def _s(v):
                        return str(v) if v is not None else None
                    safe_labs.append(LabResult(
                        test=str(l.get("test", l.get("name", "Unknown"))) if isinstance(l, dict) else str(l),
                        value=_s(l.get("value", l.get("result"))) if isinstance(l, dict) else None,
                        unit=_s(l.get("unit")) if isinstance(l, dict) else None,
                        reference_range=_s(l.get("reference_range")) if isinstance(l, dict) else None,
                        status="ABNORMAL" if (isinstance(l, dict) and l.get("abnormal", False)) else "FINAL",
                    ))
                except Exception as lab_err:
                    self.tracer.log_warning(f"Skipping malformed lab result: {lab_err}")
            self.summary.key_investigations = safe_labs

        # Hospital course
        hc = ed.get("hospital_course")
        if isinstance(hc, dict):
            self.summary.hospital_course = hc.get("narrative") or hc.get("summary") or str(hc)
        elif isinstance(hc, str):
            self.summary.hospital_course = hc
        else:
            self.summary.hospital_course = MISSING

        # Imaging
        imaging = ed.get("imaging", [])
        if isinstance(imaging, list):
            self.summary.imaging_results = [
                (i.get("description") or i.get("finding") or str(i)) if isinstance(i, dict) else str(i)
                for i in imaging
            ]
        elif isinstance(imaging, str):
            self.summary.imaging_results = [imaging]

        # Follow-up
        followup = ed.get("follow_up", [])
        if isinstance(followup, list):
            self.summary.follow_up_instructions = [str(f) for f in followup if f]
        elif isinstance(followup, dict):
            fu_list = followup.get("instructions", followup.get("follow_up", []))
            if isinstance(fu_list, list):
                self.summary.follow_up_instructions = fu_list

        # Drug interactions
        if "drug_interactions" in ed:
            di = ed["drug_interactions"]
            if isinstance(di, dict) and di.get("interactions"):
                self.summary.drug_interactions_checked = True
                self.summary.drug_interactions = di["interactions"]

        # Allergy conflicts
        if "allergy_conflicts" in ed:
            ac = ed["allergy_conflicts"]
            if isinstance(ac, dict) and ac.get("conflicts"):
                self.summary.allergy_conflicts = ac["conflicts"]

        # Pending results
        if "pending" in ed:
            pend = ed["pending"]
            if isinstance(pend, dict):
                for pr in pend.get("pending_results", []):
                    self.summary.pending_results.append(
                        f"{pr.get('test','Unknown test')}: {pr.get('status','PENDING')} — {pr.get('note','')}"
                    )

        # Fields from summary_data override
        list_fields = {"presenting_complaints", "past_medical_history", "discharge_instructions", "allergies"}
        for key in ["date_of_admission", "date_of_discharge", "discharge_condition",
                    "ward", "admitting_physician", "discharging_physician",
                    "presenting_complaints", "history_of_present_illness",
                    "past_medical_history", "discharge_instructions"]:
            if key in data:
                val = data[key]
                if key in list_fields:
                    if isinstance(val, str):
                        val = [val] if val else []
                    elif isinstance(val, dict):
                        val = [str(v) for v in val.values() if v]
                    elif not isinstance(val, list):
                        val = [str(val)] if val else []
                setattr(self.summary, key, val)

        # Mark missing critical fields
        critical_fields = {
            "patient_name": self.summary.patient_name,
            "date_of_admission": self.summary.date_of_admission,
            "principal_diagnosis": self.summary.principal_diagnosis,
            "hospital_course": self.summary.hospital_course,
            "discharge_condition": self.summary.discharge_condition,
        }
        for field, value in critical_fields.items():
            if value == MISSING or not value:
                self.summary.add_flag(
                    "MISSING_DATA",
                    f"Critical field '{field}' could not be sourced from documents.",
                    "HIGH",
                )

        return {
            "success": True,
            "message": "Discharge summary compiled.",
            "flags_count": len(self.summary.flags),
            "escalations_count": len(self.escalations_made),
        }

    # ── Main Agent Loop ───────────────────────────────────────────────────────

    def run(self) -> DischargeSummary:
        """Execute the agentic loop."""
        self.tracer.log_info(f"Starting agent for patient: {self.patient_id}")
        self.tracer.log_info(f"Patient folder: {self.patient_folder}")
        self.tracer.log_info(f"Max steps: {MAX_STEPS}")

        # Initial user message to kick off the agent
        self.messages.append({
            "role": "user",
            "content": (
                f"Please generate a complete discharge summary for patient '{self.patient_id}'. "
                f"The source documents are in the folder: {self.patient_folder}\n\n"
                f"Follow your agentic workflow: ingest PDFs → extract all clinical data → "
                f"run safety checks → compile the summary. "
                f"Remember: NEVER fabricate any clinical fact. "
                f"Flag everything you cannot source as MISSING or PENDING."
            ),
        })

        step_num = 0
        summary_compiled = False

        while step_num < MAX_STEPS:
            step_num += 1

            # ── LLM Call ──────────────────────────────────────────────
            try:
                response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=3000,
                )
            except Exception as e:
                self.tracer.log_warning(f"LLM API call failed at step {step_num}: {e}")
                # Retry once
                try:
                    time.sleep(2)
                    response = self.client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=self.messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=0.1,
                        max_tokens=3000,
                    )
                except Exception as e2:
                    self.tracer.log_warning(f"LLM retry also failed: {e2}. Terminating loop.")
                    break

            choice = response.choices[0]
            message = choice.message

            # ── Check for tool calls ───────────────────────────────────
            if not message.tool_calls:
                # No tool call — agent is done or needs a nudge
                content = message.content or ""
                self.tracer.log_info(f"Agent response (no tool call): {content[:200]}")

                if summary_compiled or "complete" in content.lower() or "done" in content.lower():
                    self.tracer.log_success("Agent finished.")
                    break

                # If the agent hasn't compiled yet, nudge it
                if step_num < MAX_STEPS - 1 and not summary_compiled:
                    self.messages.append({"role": "assistant", "content": content or "Continuing..."})
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Please continue. If you have all the data you need, call compile_discharge_summary. "
                            "If there are more extractions to do, continue extracting. "
                            "Remember to check for drug interactions and pending results before compiling."
                        ),
                    })
                    continue
                break

            # ── Execute all tool calls in this step ────────────────────
            # Add assistant message with tool calls
            self.messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                # Determine reasoning from message content
                reasoning = message.content or f"Calling {tool_name}"
                step = self.tracer.start_step(step_num, reasoning)

                # Dispatch
                result = self._dispatch_tool(tool_name, tool_args)

                # Store results in extracted_data for later use
                if tool_name == "check_drug_interactions" and isinstance(result, dict) and result.get("success"):
                    self.extracted_data["drug_interactions"] = result
                elif tool_name == "check_allergy_conflicts" and isinstance(result, dict) and result.get("success"):
                    self.extracted_data["allergy_conflicts"] = result
                elif tool_name == "lookup_pending_results" and isinstance(result, dict) and result.get("success"):
                    self.extracted_data["pending"] = result
                elif tool_name == "compile_discharge_summary":
                    summary_compiled = True

                # Determine next decision label
                next_decision = "Continue extracting data" if not summary_compiled else "Summary compiled — finish"

                # Truncate result for trace display
                result_for_trace = result
                if isinstance(result, dict) and "data" in result:
                    result_for_trace = {k: v for k, v in result.items() if k != "data"}
                    result_for_trace["data"] = "[extracted — stored internally]"

                self.tracer.complete_step(
                    action=tool_name,
                    inputs=tool_args,
                    result=result_for_trace,
                    next_decision=next_decision,
                )

                # Add tool result to messages
                result_str = json.dumps(result, default=str)
                if len(result_str) > 3000:
                    result_str = result_str[:3000] + "...[truncated]"

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        # ── Hard cap reached ───────────────────────────────────────────
        if step_num >= MAX_STEPS:
            self.tracer.log_warning(f"Hard step cap ({MAX_STEPS}) reached. Forcing summary compilation.")
            self.summary.add_flag(
                "SYSTEM",
                f"Agent reached maximum step limit ({MAX_STEPS}). Some fields may be incomplete. Clinician must verify all sections.",
                "HIGH",
            )
            # Force compile with whatever we have
            self._tool_compile_summary({"summary_data": {}})

        # ── Save outputs ───────────────────────────────────────────────
        self.tracer.print_summary()

        # Save readable summary
        summary_text = self.summary.to_readable_text()
        summary_path = self.output_dir / f"{self.patient_id}_discharge_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary_text)
        self.tracer.log_success(f"Discharge summary → {summary_path}")

        # Save JSON summary
        json_path = self.output_dir / f"{self.patient_id}_discharge_summary.json"
        with open(json_path, "w") as f:
            f.write(self.summary.to_json())
        self.tracer.log_success(f"JSON summary → {json_path}")

        # Save trace
        self.tracer.save_trace()

        return self.summary


def run_agent(patient_folder: str, patient_id: str, output_dir: str = "output") -> DischargeSummary:
    """Convenience function to run the agent."""
    agent = DischargeAgent(
        patient_folder=patient_folder,
        patient_id=patient_id,
        output_dir=output_dir,
    )
    return agent.run()
