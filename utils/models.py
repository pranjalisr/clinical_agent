"""
Discharge Summary Data Model
Structured, validated schema for the discharge summary output.
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json


MISSING = "[[MISSING - CLINICIAN REVIEW REQUIRED]]"
PENDING = "[[PENDING - AWAITING RESULT]]"
CONFLICT = "[[CONFLICT - CLINICIAN REVIEW REQUIRED]]"


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None
    change_from_admission: Optional[str] = None  # "NEW", "STOPPED", "CHANGED", "CONTINUED", "UNDOCUMENTED_CHANGE"
    flag: Optional[str] = None  # Any safety concern


class LabResult(BaseModel):
    model_config = {"coerce_numbers_to_str": True}

    test: str
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: str = "FINAL"  # "FINAL", "PENDING", "ABNORMAL"
    note: Optional[str] = None


class Conflict(BaseModel):
    field: str
    value_a: str
    source_a: str
    value_b: str
    source_b: str
    resolution: str = CONFLICT


class Flag(BaseModel):
    category: str  # "SAFETY", "MISSING_DATA", "CONFLICT", "PENDING", "MEDICATION"
    message: str
    severity: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    requires_clinician_action: bool = True


class DischargeSummary(BaseModel):
    # Metadata
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent_version: str = "1.0.0"
    status: str = "DRAFT - FOR CLINICIAN REVIEW ONLY"
    patient_id: str

    # Demographics
    patient_name: str = MISSING
    age: Optional[str] = None
    gender: Optional[str] = None
    mrn: Optional[str] = None
    blood_group: Optional[str] = None

    # Admission details
    date_of_admission: str = MISSING
    date_of_discharge: str = MISSING
    ward: Optional[str] = None
    admitting_physician: Optional[str] = None
    discharging_physician: Optional[str] = None

    # Clinical
    principal_diagnosis: str = MISSING
    secondary_diagnoses: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    allergy_note: Optional[str] = None

    # Vitals on admission
    vitals_on_admission: dict = Field(default_factory=dict)
    vitals_on_discharge: dict = Field(default_factory=dict)

    # Hospital course
    presenting_complaints: list[str] = Field(default_factory=list)
    history_of_present_illness: str = MISSING
    past_medical_history: list[str] = Field(default_factory=list)
    hospital_course: str = MISSING

    # Investigations
    key_investigations: list[LabResult] = Field(default_factory=list)
    imaging_results: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)

    # Medications
    admission_medications: list[Medication] = Field(default_factory=list)
    discharge_medications: list[Medication] = Field(default_factory=list)
    medication_reconciliation_notes: list[str] = Field(default_factory=list)

    # Discharge
    discharge_condition: str = MISSING
    discharge_instructions: list[str] = Field(default_factory=list)
    follow_up_instructions: list[str] = Field(default_factory=list)
    pending_results: list[str] = Field(default_factory=list)
    diet_instructions: Optional[str] = None

    # Safety
    conflicts_detected: list[Conflict] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    drug_interactions_checked: bool = False
    drug_interactions: list[dict] = Field(default_factory=list)
    allergy_conflicts: list[dict] = Field(default_factory=list)

    def add_flag(self, category: str, message: str, severity: str = "MEDIUM"):
        self.flags.append(Flag(
            category=category,
            message=message,
            severity=severity,
            requires_clinician_action=True,
        ))

    def to_readable_text(self) -> str:
        """Render as a clean, human-readable discharge summary document."""
        lines = []
        sep = "=" * 70

        lines.append(sep)
        lines.append("           DISCHARGE SUMMARY DRAFT")
        lines.append("         ⚠  FOR CLINICIAN REVIEW ONLY ⚠")
        lines.append(sep)
        lines.append(f"Generated: {self.generated_at}")
        lines.append(f"Status:    {self.status}")
        lines.append("")

        # ── Patient Info ──────────────────────────────────────────────
        lines.append("PATIENT INFORMATION")
        lines.append("-" * 40)
        lines.append(f"Name:           {self.patient_name}")
        if self.age:
            lines.append(f"Age/Gender:     {self.age} / {self.gender or MISSING}")
        if self.mrn:
            lines.append(f"MRN:            {self.mrn}")
        if self.blood_group:
            lines.append(f"Blood Group:    {self.blood_group}")
        lines.append(f"Admission:      {self.date_of_admission}")
        lines.append(f"Discharge:      {self.date_of_discharge}")
        if self.ward:
            lines.append(f"Ward:           {self.ward}")
        if self.admitting_physician:
            lines.append(f"Admitting Dr:   {self.admitting_physician}")
        if self.discharging_physician:
            lines.append(f"Discharging Dr: {self.discharging_physician}")
        lines.append("")

        # ── Diagnoses ─────────────────────────────────────────────────
        lines.append("DIAGNOSES")
        lines.append("-" * 40)
        lines.append(f"Principal:      {self.principal_diagnosis}")
        if self.secondary_diagnoses:
            lines.append("Secondary:")
            for dx in self.secondary_diagnoses:
                lines.append(f"  • {dx}")
        lines.append("")

        # ── Allergies ─────────────────────────────────────────────────
        lines.append("ALLERGIES")
        lines.append("-" * 40)
        if self.allergies:
            for a in self.allergies:
                lines.append(f"  • {a}")
        else:
            lines.append(f"  {MISSING}")
        if self.allergy_note:
            lines.append(f"  Note: {self.allergy_note}")
        lines.append("")

        # ── Presenting Complaints & HPI ───────────────────────────────
        lines.append("PRESENTING COMPLAINTS")
        lines.append("-" * 40)
        if self.presenting_complaints:
            for c in self.presenting_complaints:
                lines.append(f"  • {c}")
        lines.append("")

        lines.append("HISTORY OF PRESENT ILLNESS")
        lines.append("-" * 40)
        lines.append(self.history_of_present_illness)
        lines.append("")

        if self.past_medical_history:
            lines.append("PAST MEDICAL HISTORY")
            lines.append("-" * 40)
            for h in self.past_medical_history:
                lines.append(f"  • {h}")
            lines.append("")

        # ── Vitals ────────────────────────────────────────────────────
        if self.vitals_on_admission:
            lines.append("VITALS ON ADMISSION")
            lines.append("-" * 40)
            for k, v in self.vitals_on_admission.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # ── Hospital Course ───────────────────────────────────────────
        lines.append("HOSPITAL COURSE")
        lines.append("-" * 40)
        lines.append(self.hospital_course)
        lines.append("")

        # ── Investigations ────────────────────────────────────────────
        if self.key_investigations:
            lines.append("KEY INVESTIGATIONS")
            lines.append("-" * 40)
            for lab in self.key_investigations:
                status_tag = f" [{lab.status}]" if lab.status not in ("FINAL",) else ""
                val_str = f"{lab.value} {lab.unit or ''}".strip() if lab.value else PENDING
                lines.append(f"  {lab.test}: {val_str}{status_tag}")
                if lab.note:
                    lines.append(f"    → {lab.note}")
            lines.append("")

        if self.imaging_results:
            lines.append("IMAGING / RADIOLOGY")
            lines.append("-" * 40)
            for img in self.imaging_results:
                lines.append(f"  • {img}")
            lines.append("")

        if self.procedures:
            lines.append("PROCEDURES PERFORMED")
            lines.append("-" * 40)
            for p in self.procedures:
                lines.append(f"  • {p}")
            lines.append("")

        # ── Medications ───────────────────────────────────────────────
        lines.append("DISCHARGE MEDICATIONS")
        lines.append("-" * 40)
        if self.discharge_medications:
            for med in self.discharge_medications:
                parts = [med.name]
                if med.dose:
                    parts.append(med.dose)
                if med.frequency:
                    parts.append(med.frequency)
                if med.duration:
                    parts.append(f"× {med.duration}")
                med_line = "  • " + " | ".join(parts)
                if med.change_from_admission:
                    med_line += f"  [{med.change_from_admission}]"
                if med.flag:
                    med_line += f"\n    ⚠ {med.flag}"
                lines.append(med_line)
        else:
            lines.append(f"  {MISSING}")

        if self.medication_reconciliation_notes:
            lines.append("\n  Reconciliation Notes:")
            for note in self.medication_reconciliation_notes:
                lines.append(f"    ⚠ {note}")
        lines.append("")

        # ── Discharge Condition & Instructions ────────────────────────
        lines.append("DISCHARGE CONDITION")
        lines.append("-" * 40)
        lines.append(f"  {self.discharge_condition}")
        lines.append("")

        if self.discharge_instructions:
            lines.append("DISCHARGE INSTRUCTIONS")
            lines.append("-" * 40)
            for instr in self.discharge_instructions:
                lines.append(f"  • {instr}")
            lines.append("")

        if self.follow_up_instructions:
            lines.append("FOLLOW-UP")
            lines.append("-" * 40)
            for fu in self.follow_up_instructions:
                lines.append(f"  • {fu}")
            lines.append("")

        if self.pending_results:
            lines.append("PENDING RESULTS")
            lines.append("-" * 40)
            for pr in self.pending_results:
                lines.append(f"  ⏳ {pr}")
            lines.append("")

        # ── Safety Flags & Conflicts ──────────────────────────────────
        if self.conflicts_detected or self.flags or self.drug_interactions or self.allergy_conflicts:
            lines.append(sep)
            lines.append("          ⚠  CLINICIAN REVIEW REQUIRED  ⚠")
            lines.append(sep)

        if self.flags:
            lines.append("\nFLAGS FOR REVIEW")
            lines.append("-" * 40)
            for flag in self.flags:
                severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(flag.severity, "⚠")
                lines.append(f"  {severity_icon} [{flag.severity}] [{flag.category}]")
                lines.append(f"     {flag.message}")
            lines.append("")

        if self.conflicts_detected:
            lines.append("CONFLICTS DETECTED")
            lines.append("-" * 40)
            for conflict in self.conflicts_detected:
                lines.append(f"  ⚡ Field: {conflict.field}")
                lines.append(f"     Source A ({conflict.source_a}): {conflict.value_a}")
                lines.append(f"     Source B ({conflict.source_b}): {conflict.value_b}")
                lines.append(f"     → {conflict.resolution}")
            lines.append("")

        if self.drug_interactions:
            lines.append("DRUG INTERACTION ALERTS")
            lines.append("-" * 40)
            for ix in self.drug_interactions:
                lines.append(f"  ⚠ {ix.get('drug_a','?')} ↔ {ix.get('drug_b','?')}: {ix.get('severity','?')}")
                lines.append(f"     {ix.get('description','')}")
                lines.append(f"     Recommendation: {ix.get('recommendation','')}")
            lines.append("")

        if self.allergy_conflicts:
            lines.append("ALLERGY CONFLICT ALERTS")
            lines.append("-" * 40)
            for ac in self.allergy_conflicts:
                lines.append(f"  🔴 ALLERGY: {ac.get('allergy','?')} → Medication: {ac.get('medication','?')}")
                lines.append(f"     {ac.get('recommendation','')}")
            lines.append("")

        lines.append(sep)
        lines.append("END OF DISCHARGE SUMMARY DRAFT")
        lines.append(f"This document was auto-generated and must be reviewed and signed by a clinician before use.")
        lines.append(sep)

        return "\n".join(lines)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
