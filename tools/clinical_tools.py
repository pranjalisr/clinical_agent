"""
Drug Interaction & Safety Tools (Mocked)

In production these would call a real drug database (e.g. DrugBank, OpenFDA).
Here they are realistic mocks that the agent decides WHEN to call.
"""

import random
import time

# Mock drug interaction database
_KNOWN_INTERACTIONS = {
    frozenset(["warfarin", "aspirin"]): {
        "severity": "MAJOR",
        "description": "Concurrent use significantly increases bleeding risk.",
        "recommendation": "Avoid combination or monitor INR closely.",
    },
    frozenset(["metformin", "contrast dye"]): {
        "severity": "MAJOR",
        "description": "Risk of contrast-induced nephropathy and lactic acidosis.",
        "recommendation": "Hold metformin 48h before and after contrast administration.",
    },
    frozenset(["ofloxacin", "metformin"]): {
        "severity": "MODERATE",
        "description": "Fluoroquinolones can cause hypoglycaemia or hyperglycaemia in diabetic patients.",
        "recommendation": "Monitor blood glucose closely.",
    },
    frozenset(["pantoprazole", "clopidogrel"]): {
        "severity": "MODERATE",
        "description": "PPIs may reduce antiplatelet effect of clopidogrel.",
        "recommendation": "Consider alternative PPI or reassess need.",
    },
    frozenset(["loperamide", "metronidazole"]): {
        "severity": "MINOR",
        "description": "Loperamide may mask symptoms being treated by metronidazole.",
        "recommendation": "Use with caution; reassess if symptoms worsen.",
    },
}

# Mock allergy reaction database
_ALLERGY_RISK = {
    "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "cephalosporin"],
    "sulfa": ["sulfamethoxazole", "furosemide", "thiazide"],
    "nsaid": ["ibuprofen", "naproxen", "mefenamic acid", "meftal"],
    "contrast": ["iodine", "contrast dye"],
}


def check_drug_interactions(medications: list[str]) -> dict:
    """
    Check for drug-drug interactions among a list of medications.
    Simulates a real drug interaction API call with occasional timeouts.
    """
    result = {
        "success": False,
        "interactions": [],
        "checked_count": len(medications),
        "error": None,
    }

    # Simulate occasional tool failure (10% chance)
    if random.random() < 0.10:
        result["error"] = "Drug interaction service temporarily unavailable (timeout)"
        return result

    # Simulate latency
    time.sleep(0.1)

    meds_lower = [m.lower().strip() for m in medications]

    found_interactions = []
    for i, med1 in enumerate(meds_lower):
        for med2 in meds_lower[i + 1:]:
            key = frozenset([med1, med2])
            if key in _KNOWN_INTERACTIONS:
                found_interactions.append({
                    "drug_a": med1,
                    "drug_b": med2,
                    **_KNOWN_INTERACTIONS[key],
                })
            else:
                # Check partial matches
                for db_key, interaction in _KNOWN_INTERACTIONS.items():
                    db_list = list(db_key)
                    if any(d in med1 for d in db_list) and any(d in med2 for d in db_list):
                        found_interactions.append({
                            "drug_a": med1,
                            "drug_b": med2,
                            **interaction,
                        })

    result["success"] = True
    result["interactions"] = found_interactions
    return result


def check_allergy_conflicts(allergies: list[str], medications: list[str]) -> dict:
    """
    Check if any prescribed medication conflicts with known allergies.
    """
    result = {
        "success": False,
        "conflicts": [],
        "error": None,
    }

    if random.random() < 0.05:
        result["error"] = "Allergy check service unavailable"
        return result

    time.sleep(0.05)

    allergies_lower = [a.lower() for a in allergies]
    meds_lower = [m.lower() for m in medications]

    conflicts = []
    for allergy in allergies_lower:
        cross_reactors = _ALLERGY_RISK.get(allergy, [allergy])
        for med in meds_lower:
            for reactor in cross_reactors:
                if reactor in med or med in reactor:
                    conflicts.append({
                        "allergy": allergy,
                        "medication": med,
                        "severity": "HIGH",
                        "recommendation": f"CONTRAINDICATED: Patient has known {allergy} allergy. Prescribing {med} requires urgent clinician review.",
                    })

    result["success"] = True
    result["conflicts"] = conflicts
    return result


def escalate_for_clinician_review(reason: str, details: dict) -> dict:
    """
    Formally escalate an issue to clinician review.
    In production this would create a task/alert in the EHR.
    """
    time.sleep(0.05)
    return {
        "success": True,
        "escalation_id": f"ESC-{int(time.time())}",
        "reason": reason,
        "details": details,
        "status": "FLAGGED_FOR_CLINICIAN_REVIEW",
        "message": f"Escalation logged. Clinician review required: {reason}",
    }


def lookup_pending_results(patient_id: str) -> dict:
    """
    Check if any lab/culture results are still pending.
    Mocked: returns realistic pending result data.
    """
    if random.random() < 0.10:
        return {"success": False, "error": "Lab system unavailable"}

    return {
        "success": True,
        "patient_id": patient_id,
        "pending_results": [
            {
                "test": "Urine Culture & Sensitivity",
                "ordered_date": "UNKNOWN",
                "expected_date": "2-3 business days from collection",
                "status": "PENDING",
                "note": "Sample collected during admission; result not yet available in records",
            }
        ],
    }
